"""Lightweight FastAPI serverless API for Renju tentative-first outcome prediction."""

from __future__ import annotations

import gzip
import json
import math
import re
import sys
from bisect import bisect_left
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nn.features import (
    build_input_planes,
    canonicalize_d4,
    coord_to_str,
    invert_d4_coord,
    parse_coord,
    tf_current_color_at_move,
)

BUNDLE_PATH = ROOT_DIR / "artifacts" / "vercel_bundle.json.gz"
ONNX_PATH = ROOT_DIR / "artifacts" / "renju_dualnet.onnx"
VOCAB_PATH = ROOT_DIR / "artifacts" / "player_vocab.json"

app = FastAPI(
    title="Renju Outcome Predictor API",
    description="Strictly causal Glicko-1 + H2H + LightGBM & DualNet (Policy + Value) predictor",
    version="2.0.0",
)

# The static site (renjupgc.github.io) calls this API straight from the browser.
# Without CORS it had to route every request through a third-party reader proxy,
# which is what used to time out on mobile networks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)


_STATE: Optional[Dict[str, Any]] = None
_ONNX_SESSION: Optional[ort.InferenceSession] = None
_PLAYER_VOCAB: Optional[Dict[int, int]] = None


def _get_player_vocab() -> Dict[int, int]:
    global _PLAYER_VOCAB
    if _PLAYER_VOCAB is None:
        if VOCAB_PATH.exists():
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                _PLAYER_VOCAB = {int(k): int(v) for k, v in json.load(f).items()}
        else:
            _PLAYER_VOCAB = {}
    return _PLAYER_VOCAB


def _get_onnx_session() -> ort.InferenceSession:
    global _ONNX_SESSION
    if _ONNX_SESSION is None:
        if not ONNX_PATH.exists():
            raise HTTPException(status_code=503, detail="Neural network ONNX model not found.")
        _ONNX_SESSION = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return _ONNX_SESSION


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
        "source": "prematch",
        "model": "lightgbm",
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
            "tf_win": round(probs[0], 6),
            "draw": round(probs[1], 6),
            "ts_win": round(probs[2], 6),
        },
    }


class PredictRequest(BaseModel):
    time: str
    rule: Union[int, str]
    tf: Union[int, str]
    ts: Union[int, str]


class PredictBoardRequest(BaseModel):
    time: str
    rule: Union[int, str]
    tf: Union[int, str]
    ts: Union[int, str]
    moves: Union[str, List[str]] = ""
    tf_color: Optional[Union[int, str]] = None
    swap: str = ""
    top_k: int = 5


def _run_predict_board(
    ts: str,
    rule: Union[int, str],
    tf: Union[int, str],
    ts_p: Union[int, str],
    moves: Union[str, List[str]] = "",
    tf_color: Optional[Union[int, str]] = None,
    swap: str = "",
    top_k: int = 5,
) -> dict:
    state = _load_state()
    date_str = _normalize_timestamp(ts)
    rule_id = _resolve_rule(state, rule)
    tf_id = _resolve_player(state, tf)
    ts_id = _resolve_player(state, ts_p)

    hist = state["history"]
    tf_r, tf_rd = _get_rating_and_rd(hist, tf_id, date_str)
    ts_r, ts_rd = _get_rating_and_rd(hist, ts_id, date_str)
    h2h_g, h2h_sr, h2h_dr = _get_h2h(hist, tf_id, ts_id, date_str)

    if isinstance(moves, str):
        move_tokens = [m for m in re.split(r"[\s,]+", moves.strip()) if m]
    else:
        move_tokens = [str(m).strip() for m in moves if str(m).strip()]

    # No position on the board yet: there is nothing for the policy/value net to read,
    # so fall back to the pre-match (赛前) LightGBM win-rate model.
    if not move_tokens:
        rule_cat_idx = state["rule_cat_to_idx"].get(rule_id, -1)
        x = (rule_cat_idx, tf_r, ts_r, tf_r - ts_r, tf_rd, ts_rd, h2h_g, h2h_sr, h2h_dr)
        probs = _predict_probs(state["trees_by_class"], x)
        value = {
            "tf_win": round(float(probs[0]), 6),
            "draw": round(float(probs[1]), 6),
            "ts_win": round(float(probs[2]), 6),
        }
        return {
            "timestamp": date_str,
            "source": "prematch",
            "model": "lightgbm",
            "board_empty": True,
            "note": "空棋盘，暂无局面：已改用赛前胜率预测（等级分 + 交手记录）。",
            "rule": {"id": rule_id, "name": state["rules"][rule_id]},
            "move_count": 0,
            "next_to_move": "black",
            "policy_top_moves": [],
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
                "rating_diff": round(tf_r - ts_r, 2),
                "tf_rd": round(tf_rd, 2),
                "ts_rd": round(ts_rd, 2),
                "h2h_games": h2h_g,
                "h2h_score_rate": round(h2h_sr, 4),
                "h2h_draw_rate": round(h2h_dr, 4),
            },
            "prematch": dict(value),
            "value": value,
        }

    board = np.zeros((15, 15), dtype=np.int8)
    parsed: List[Tuple[int, int]] = []
    for idx, tok in enumerate(move_tokens):
        xy = parse_coord(tok)
        if xy is None:
            raise HTTPException(status_code=400, detail=f"Invalid move coordinate: '{tok}'")
        x, y = xy
        if board[y, x] != 0:
            raise HTTPException(status_code=400, detail=f"Duplicate stone at '{tok}'")
        board[y, x] = 1 if (idx % 2 == 0) else 2
        parsed.append((x, y))

    move_idx = len(parsed)
    next_color = 1 if (move_idx % 2 == 0) else 2

    if tf_color is not None and str(tf_color).strip() != "":
        c_str = str(tf_color).strip().lower()
        if c_str in ("1", "b", "black", "黑"):
            tf_cur = 1
        elif c_str in ("2", "w", "white", "白"):
            tf_cur = 2
        else:
            raise HTTPException(status_code=400, detail=f"Invalid tf_color: '{tf_color}'")
    else:
        tf_cur = tf_current_color_at_move(rule_id, swap, move_idx)

    last_m = parsed[-1] if move_idx >= 1 else None
    prev2_m = parsed[-2] if move_idx >= 2 else None

    cb, clm, cp2, _, csym, _ = canonicalize_d4(board, last_m, prev2_m)

    planes, legal = build_input_planes(
        board=cb,
        next_color=next_color,
        tf_current_color=tf_cur,
        rule_id=rule_id,
        move_idx=move_idx,
        last_move=clm,
        prev2_move=cp2,
        tf_rating=tf_r,
        ts_rating=ts_r,
        tf_rd=tf_rd,
        ts_rd=ts_rd,
        h2h_games=h2h_g,
        h2h_score_rate=h2h_sr,
        h2h_draw_rate=h2h_dr,
    )

    p_vocab = _get_player_vocab()
    tf_vid = p_vocab.get(tf_id, 0)
    ts_vid = p_vocab.get(ts_id, 0)
    if next_color == tf_cur:
        mover_vid, opp_vid = tf_vid, ts_vid
    else:
        mover_vid, opp_vid = ts_vid, tf_vid

    sess = _get_onnx_session()
    p_logits, v_logits = sess.run(
        None,
        {
            "input": planes[None, ...],
            "mover_id": np.array([mover_vid], dtype=np.int64),
            "opp_id": np.array([opp_vid], dtype=np.int64),
        },
    )
    p_logits = p_logits[0]
    p_logits = np.where(legal, p_logits, -1e9)
    p_max = float(np.max(p_logits))
    p_exp = np.exp(p_logits - p_max)
    p_probs = p_exp / float(np.sum(p_exp))

    v_raw = v_logits[0]
    v_max = float(np.max(v_raw))
    v_exp = np.exp(v_raw - v_max)
    v_probs = v_exp / float(np.sum(v_exp))

    k = max(1, min(int(top_k), 25))
    top_indices = np.argsort(-p_probs)[:k]
    policy_top = []
    for idx in top_indices:
        if legal[idx]:
            ox, oy = invert_d4_coord(int(idx % 15), int(idx // 15), csym)
            policy_top.append(
                {
                    "move": coord_to_str(ox, oy),
                    "prob": round(float(p_probs[idx]), 6),
                }
            )

    return {
        "timestamp": date_str,
        "source": "dualnet",
        "board_empty": False,
        "rule": {"id": rule_id, "name": state["rules"][rule_id]},
        "move_count": move_idx,
        "next_to_move": "black" if next_color == 1 else "white",
        "tf_current_color": "black" if tf_cur == 1 else "white",
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
        "policy_top_moves": policy_top,
        "value": {
            "tf_win": round(float(v_probs[0]), 6),
            "draw": round(float(v_probs[1]), 6),
            "ts_win": round(float(v_probs[2]), 6),
        },
    }


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


@app.get("/predict_board")
def predict_board_get(
    time: str = Query(..., description="Timestamp YYYY-MM-DD"),
    rule: str = Query(..., description="Rule ID or name"),
    tf: str = Query(..., description="Tentative first player ID or name"),
    ts: str = Query(..., description="Tentative second player ID or name"),
    moves: str = Query("", description="Space or comma-separated moves, e.g. 'h8 i9 i8'"),
    tf_color: Optional[str] = Query(None, description="Current color of 假先: black or white"),
    swap: str = Query("", description="Optional swap sequence if tf_color omitted"),
    top_k: int = Query(5, ge=1, le=25, description="Top-K policy moves"),
) -> dict:
    return _run_predict_board(time, rule, tf, ts, moves=moves, tf_color=tf_color, swap=swap, top_k=top_k)


@app.post("/predict_board")
def predict_board_post(req: PredictBoardRequest) -> dict:
    return _run_predict_board(
        req.time,
        req.rule,
        req.tf,
        req.ts,
        moves=req.moves,
        tf_color=req.tf_color,
        swap=req.swap,
        top_k=req.top_k,
    )
