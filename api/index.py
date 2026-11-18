"""Lightweight FastAPI serverless API for Renju tentative-first outcome prediction."""

from __future__ import annotations

import gzip
import json
import math
import re
from bisect import bisect_left
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "vercel_bundle.json.gz"

app = FastAPI(
    title="Renju Outcome Predictor API",
    description="Strictly causal Glicko-1 + H2H + LightGBM predictor for tentative_first_win / draw / tentative_second_win",
    version="1.0.0",
)

_STATE: Optional[Dict[str, Any]] = None


def _load_state() -> Dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE

    with gzip.open(BUNDLE_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Pre-index rule names & categories
    rules: Dict[int, str] = {int(k): v for k, v in data["rules"].items()}
    rule_by_name: Dict[str, int] = {v.strip().lower(): k for k, v in rules.items()}
    rule_cat_to_idx: Dict[int, int] = {
        int(rid): idx for idx, rid in enumerate(data["rule_categories"])
    }

    # Pre-index players
    players: Dict[int, dict] = {int(k): v for k, v in data["players"].items()}
    players_by_disp: Dict[str, int] = {}
    players_by_name: Dict[str, List[int]] = {}
    for pid, info in players.items():
        if info.get("d"):
            players_by_disp[info["d"].strip()] = pid
        for cand in {
            info.get("n", "").strip().lower(),
            info.get("rn", "").strip().lower(),
            info.get("nn", "").strip().lower(),
        }:
            if cand:
                players_by_name.setdefault(cand, []).append(pid)

    # Pre-parse LightGBM trees by class index (tree_index % num_class)
    num_class = data["booster"]["num_class"]
    trees_by_class: List[List[dict]] = [[] for _ in range(num_class)]
    for t_info in data["booster"]["tree_info"]:
        cls_idx = t_info["tree_index"] % num_class
        trees_by_class[cls_idx].append(t_info["tree_structure"])

    _STATE = {
        "rules": rules,
        "rule_by_name": rule_by_name,
        "rule_cat_to_idx": rule_cat_to_idx,
        "players": players,
        "players_by_disp": players_by_disp,
        "players_by_name": players_by_name,
        "history": data["history"],
        "trees_by_class": trees_by_class,
        "metrics": data["metrics"],
    }
    return _STATE


def _eval_tree(node: dict, x: Tuple[int, float, float, float, float, float, int, float, float]) -> float:
    """Evaluate a single LightGBM tree structure in pure Python."""
    while "leaf_value" not in node:
        fidx = node["split_feature"]
        val = x[fidx]
        dtype = node["decision_type"]
        if dtype == "==":
            # Categorical split: threshold is "cat1||cat2||..."
            cats = node.get("_cat_set")
            if cats is None:
                cats = {int(c) for c in str(node["threshold"]).split("||")}
                node["_cat_set"] = cats
            go_left = int(val) in cats
        else:
            # Numerical split: "<="
            go_left = val <= node["threshold"]
        node = node["left_child"] if go_left else node["right_child"]
    return float(node["leaf_value"])


def _predict_probs(
    trees_by_class: List[List[dict]],
    x: Tuple[int, float, float, float, float, float, int, float, float],
) -> List[float]:
    raw_scores = [sum(_eval_tree(t, x) for t in cls_trees) for cls_trees in trees_by_class]
    m = max(raw_scores)
    exps = [math.exp(s - m) for s in raw_scores]
    denom = sum(exps)
    return [e / denom for e in exps]


def _normalize_timestamp(ts: str) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", str(ts).strip())
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp '{ts}'. Expected 'YYYY-MM-DD'.",
        )
    return m.group(1)


def _resolve_rule(state: Dict[str, Any], rule: Union[int, str]) -> int:
    rules = state["rules"]
    if isinstance(rule, int) or (isinstance(rule, str) and rule.strip().isdigit()):
        rid = int(rule)
        if rid in rules:
            return rid
        raise HTTPException(status_code=400, detail=f"Unknown rule ID: {rid}")
    key = str(rule).strip().lower()
    if key in state["rule_by_name"]:
        return state["rule_by_name"][key]
    matches = [rid for name, rid in state["rule_by_name"].items() if key in name]
    if len(matches) == 1:
        return matches[0]
    raise HTTPException(status_code=400, detail=f"Unknown or ambiguous rule '{rule}'.")


def _resolve_player(state: Dict[str, Any], player: Union[int, str]) -> int:
    players = state["players"]
    if isinstance(player, int) or (isinstance(player, str) and player.strip().isdigit()):
        pid = int(player)
        if pid in players:
            return pid
        s_pid = str(player).strip()
        if s_pid in state["players_by_disp"]:
            return state["players_by_disp"][s_pid]
        raise HTTPException(status_code=400, detail=f"Unknown player ID: {player}")

    key = str(player).strip().lower()
    ids = state["players_by_name"].get(key)
    if not ids:
        raise HTTPException(status_code=400, detail=f"Unknown player '{player}'.")
    if len(ids) == 1:
        return ids[0]
    opts = ", ".join(f"{pid} ({players[pid]['n']})" for pid in ids[:10])
    raise HTTPException(
        status_code=400,
        detail=f"Multiple players match '{player}': {opts}. Pass player ID.",
    )


def _get_rating_and_rd(hist: dict, pid: int, ts: str) -> Tuple[float, float]:
    spid = str(pid)
    p_dates = hist["dates"].get(spid)
    if not p_dates:
        return hist["initial_rating"], hist["initial_rd"]
    idx = bisect_left(p_dates, ts)
    if idx == 0:
        return hist["initial_rating"], hist["initial_rd"]
    r = hist["ratings"][spid][idx - 1]
    rd = hist["rds"][spid][idx - 1]
    dt = date.fromisoformat(ts).toordinal() - date.fromisoformat(p_dates[idx - 1]).toordinal()
    if dt > 0:
        rd = min(math.sqrt(rd * rd + (hist["c_per_day"] ** 2) * dt), hist["max_rd"])
    return r, rd


def _get_h2h(hist: dict, tf_id: int, ts_id: int, ts: str) -> Tuple[int, float, float]:
    key = f"{tf_id},{ts_id}" if tf_id < ts_id else f"{ts_id},{tf_id}"
    dlist = hist["h2h_dates"].get(key)
    if not dlist:
        return 0, 0.5, 0.08
    idx = bisect_left(dlist, ts)
    if idx == 0:
        return 0, 0.5, 0.08
    n_h2h = idx
    min_pts = hist["h2h_cum_pts"][key][idx - 1]
    draws = hist["h2h_cum_draws"][key][idx - 1]
    tf_pts = min_pts if tf_id < ts_id else (n_h2h - min_pts)
    return n_h2h, (tf_pts + 1.0) / (n_h2h + 2.0), (draws + 0.16) / (n_h2h + 2.0)


def _run_predict(ts: str, rule: Union[int, str], tf: Union[int, str], ts_p: Union[int, str]) -> dict:
    state = _load_state()
    date_str = _normalize_timestamp(ts)
    rule_id = _resolve_rule(state, rule)
    tf_id = _resolve_player(state, tf)
    ts_id = _resolve_player(state, ts_p)

    hist = state["history"]
    tf_r, tf_rd = _get_rating_and_rd(hist, tf_id, date_str)
    ts_r, ts_rd = _get_rating_and_rd(hist, ts_id, date_str)
    r_diff = tf_r - ts_r
    h2h_games, h2h_score, h2h_draw = _get_h2h(hist, tf_id, ts_id, date_str)

    rule_cat_idx = state["rule_cat_to_idx"].get(rule_id, -1)
    x = (rule_cat_idx, tf_r, ts_r, r_diff, tf_rd, ts_rd, h2h_games, h2h_score, h2h_draw)
    probs = _predict_probs(state["trees_by_class"], x)

    return {
        "timestamp": date_str,
        "rule": {"id": rule_id, "name": state["rules"][rule_id]},
        "tentative_first": {
            "id": tf_id,
            "name": state["players"][tf_id]["n"],
            "rating": round(tf_r, 2),
            "rd": round(tf_rd, 2),
        },
        "tentative_second": {
            "id": ts_id,
            "name": state["players"][ts_id]["n"],
            "rating": round(ts_r, 2),
            "rd": round(ts_rd, 2),
        },
        "features": {
            "rule": rule_id,
            "tf_rating": round(tf_r, 2),
            "ts_rating": round(ts_r, 2),
            "rating_diff": round(r_diff, 2),
            "tf_rd": round(tf_rd, 2),
            "ts_rd": round(ts_rd, 2),
            "h2h_games": h2h_games,
            "h2h_score_rate": round(h2h_score, 4),
            "h2h_draw_rate": round(h2h_draw, 4),
        },
        "probabilities": {
            "tentative_first_win": round(probs[0], 6),
            "draw": round(probs[1], 6),
            "tentative_second_win": round(probs[2], 6),
        },
    }


class PredictRequest(BaseModel):
    time: str
    rule: Union[int, str]
    tf: Union[int, str]
    ts: Union[int, str]


@app.get("/")
@app.get("/health")
def health() -> dict:
    state = _load_state()
    return {
        "status": "ok",
        "rules_count": len(state["rules"]),
        "players_count": len(state["players"]),
        "metrics": state["metrics"],
        "usage": {
            "GET": "/predict?time=2026-11-17&rule=Taraguchi-10&tf=Ando%20Meritee&ts=2",
            "POST": "/predict",
            "rules": "/rules",
            "players": "/players?q=Ando",
        },
    }


@app.get("/rules")
def list_rules() -> List[dict]:
    state = _load_state()
    return [{"id": k, "name": v} for k, v in sorted(state["rules"].items())]


@app.get("/players")
def search_players(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> List[dict]:
    state = _load_state()
    ql = q.strip().lower()
    res = []
    for pid, info in state["players"].items():
        if ql in info["n"].lower() or ql in info["rn"].lower() or ql == str(pid) or ql == info["d"].lower():
            res.append({"id": pid, "disp_id": info["d"], "name": info["n"], "country": info["c"]})
            if len(res) >= limit:
                break
    return res


@app.get("/predict")
def predict_get(
    time: str = Query(..., description="Timestamp YYYY-MM-DD"),
    rule: str = Query(..., description="Rule ID or name"),
    tf: str = Query(..., description="Tentative first player ID or name"),
    ts: str = Query(..., description="Tentative second player ID or name"),
) -> dict:
    return _run_predict(time, rule, tf, ts)


@app.post("/predict")
def predict_post(req: PredictRequest) -> dict:
    return _run_predict(req.time, req.rule, req.tf, req.ts)
