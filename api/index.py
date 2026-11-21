"""Unified Renju prediction API: pre-match and in-game share one model and one endpoint.

Model (ONNX):  boards (32,15,15) + tf/ts identity embeddings
               -> 226-way policy (225 board points + swap) and P(tf_win, draw, ts_win)
"""

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
    SWAP_ACTION,
    build_input_planes,
    canonicalize_d4,
    coord_to_str,
    invert_d4_coord,
    is_swap_allowed,
    parse_coord,
    tf_current_color_at_move,
)

BUNDLE_PATH = ROOT_DIR / "artifacts" / "vercel_bundle.json.gz"
ONNX_PATH = ROOT_DIR / "artifacts" / "renju_unified.onnx"
VOCAB_PATH = ROOT_DIR / "artifacts" / "player_vocab.json"

app = FastAPI(
    title="Renju Unified Predictor API",
    description="One model for pre-match and in-game: win/draw/loss probabilities and next move (including swap).",
    version="3.0.0",
)

# The static site calls this API straight from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

_STATE: Optional[Dict[str, Any]] = None
_SESSION: Optional[ort.InferenceSession] = None
_VOCAB: Optional[Dict[int, int]] = None


def _get_vocab() -> Dict[int, int]:
    global _VOCAB
    if _VOCAB is None:
        if VOCAB_PATH.exists():
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                _VOCAB = {int(k): int(v) for k, v in json.load(f).items()}
        else:
            _VOCAB = {}
    return _VOCAB


def _get_session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        if not ONNX_PATH.exists():
            raise HTTPException(status_code=503, detail="Unified model ONNX not found.")
        _SESSION = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return _SESSION


def _load_state() -> Dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE

    with gzip.open(BUNDLE_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    rules: Dict[int, str] = {int(k): v for k, v in data["rules"].items()}
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

    _STATE = {
        "rules": rules,
        "rule_by_name": {v.strip().lower(): k for k, v in rules.items()},
        "players": players,
        "players_by_disp": players_by_disp,
        "players_by_name": players_by_name,
        "history": data["history"],
        "metrics": data.get("metrics", {}),
    }
    return _STATE


def _normalize_timestamp(ts: str) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", str(ts).strip())
    if not m:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp '{ts}'. Expected 'YYYY-MM-DD'.")
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


def _parse_moves(moves: Union[str, List[str]]) -> List[Tuple[int, int]]:
    if isinstance(moves, str):
        tokens = [m for m in re.split(r"[\s,]+", moves.strip()) if m]
    else:
        tokens = [str(m).strip() for m in moves if str(m).strip()]
    parsed: List[Tuple[int, int]] = []
    for tok in tokens:
        xy = parse_coord(tok)
        if xy is None:
            raise HTTPException(status_code=400, detail=f"Invalid move coordinate: '{tok}'")
        parsed.append(xy)
    return parsed


def _color_from_value(raw: Optional[Union[int, str]]) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "":
        return None
    if s in ("1", "b", "black", "黑"):
        return 1
    if s in ("2", "w", "white", "白"):
        return 2
    raise HTTPException(status_code=400, detail=f"Invalid tf_color: '{raw}'")


def _run_predict(
    time_str: str,
    rule: Union[int, str],
    tf: Union[int, str],
    ts_player: Union[int, str],
    moves: Union[str, List[str]] = "",
    tf_color: Optional[Union[int, str]] = None,
    swap: str = "",
    top_k: int = 5,
) -> dict:
    state = _load_state()
    date_str = _normalize_timestamp(time_str)
    rule_id = _resolve_rule(state, rule)
    tf_id = _resolve_player(state, tf)
    ts_id = _resolve_player(state, ts_player)

    hist = state["history"]
    tf_r, tf_rd = _get_rating_and_rd(hist, tf_id, date_str)
    ts_r, ts_rd = _get_rating_and_rd(hist, ts_id, date_str)
    h2h_g, h2h_sr, h2h_dr = _get_h2h(hist, tf_id, ts_id, date_str)

    parsed = _parse_moves(moves)
    board = np.zeros((15, 15), dtype=np.int8)
    for idx, (x, y) in enumerate(parsed):
        if board[y, x] != 0:
            raise HTTPException(status_code=400, detail=f"Duplicate stone at move {idx + 1}")
        board[y, x] = 1 if (idx % 2 == 0) else 2

    move_idx = len(parsed)
    next_color = 1 if (move_idx % 2 == 0) else 2

    tf_cur = _color_from_value(tf_color)
    if tf_cur is None:
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

    vocab = _get_vocab()
    sess = _get_session()
    p_logits, v_logits = sess.run(
        None,
        {
            "input": planes[None, ...],
            "tf_id": np.array([vocab.get(tf_id, 0)], dtype=np.int64),
            "ts_id": np.array([vocab.get(ts_id, 0)], dtype=np.int64),
        },
    )
    p_logits = p_logits[0].astype(np.float64)
    p_logits = np.where(legal, p_logits, -1e9)
    p_exp = np.exp(p_logits - float(np.max(p_logits)))
    p_probs = p_exp / float(np.sum(p_exp))

    v_raw = v_logits[0].astype(np.float64)
    v_exp = np.exp(v_raw - float(np.max(v_raw)))
    v_probs = v_exp / float(np.sum(v_exp))

    k = max(1, min(int(top_k), 20))
    top = np.argsort(-p_probs)[: k + 1]
    policy: List[dict] = []
    for idx in top:
        idx = int(idx)
        if len(policy) >= k:
            break
        if idx == SWAP_ACTION:
            if not is_swap_allowed(move_idx):
                continue
            policy.append({"move": "swap", "prob": round(float(p_probs[idx]), 6)})
        else:
            ox, oy = invert_d4_coord(idx % 15, idx // 15, csym)
            policy.append({"move": coord_to_str(ox, oy), "prob": round(float(p_probs[idx]), 6)})

    return {
        "timestamp": date_str,
        "source": "unified",
        "board_empty": move_idx == 0,
        "rule": {"id": rule_id, "name": state["rules"][rule_id]},
        "move_count": move_idx,
        "next_to_move": "black" if next_color == 1 else "white",
        "tf_current_color": "black" if tf_cur == 1 else "white",
        "swap_available": bool(is_swap_allowed(move_idx)),
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
        "value": {
            "tf_win": round(float(v_probs[0]), 6),
            "draw": round(float(v_probs[1]), 6),
            "ts_win": round(float(v_probs[2]), 6),
        },
        "policy_top_moves": policy,
    }


class PredictRequest(BaseModel):
    time: str
    rule: Union[int, str]
    tf: Union[int, str]
    ts: Union[int, str]
    moves: Union[str, List[str]] = ""
    tf_color: Optional[Union[int, str]] = None
    swap: str = ""
    top_k: int = 5


@app.get("/")
@app.get("/health")
def health() -> dict:
    state = _load_state()
    return {
        "status": "ok",
        "model": "unified-policy-value",
        "actions": 226,
        "rules_count": len(state["rules"]),
        "players_count": len(state["players"]),
        "metrics": state["metrics"],
        "usage": {
            "GET": "/predict?time=2026-11-21&rule=Taraguchi-10&tf=Ando%20Meritee&ts=2&moves=h8%20i9",
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
    moves: str = Query("", description="Space-separated moves; empty = pre-match"),
    tf_color: Optional[str] = Query(None, description="Current colour of 假先: black/white"),
    swap: str = Query("", description="Opening swap sequence, used only if tf_color is omitted"),
    top_k: int = Query(5, ge=1, le=20),
) -> dict:
    return _run_predict(time, rule, tf, ts, moves=moves, tf_color=tf_color, swap=swap, top_k=top_k)


@app.post("/predict")
def predict_post(req: PredictRequest) -> dict:
    return _run_predict(req.time, req.rule, req.tf, req.ts, req.moves, req.tf_color, req.swap, req.top_k)


@app.get("/predict_board")
def predict_board_get(
    time: str = Query(...),
    rule: str = Query(...),
    tf: str = Query(...),
    ts: str = Query(...),
    moves: str = Query(""),
    tf_color: Optional[str] = Query(None),
    swap: str = Query(""),
    top_k: int = Query(5, ge=1, le=20),
) -> dict:
    """Deprecated alias kept for old links: same unified model, same result."""
    return _run_predict(time, rule, tf, ts, moves=moves, tf_color=tf_color, swap=swap, top_k=top_k)


@app.post("/predict_board")
def predict_board_post(req: PredictRequest) -> dict:
    return _run_predict(req.time, req.rule, req.tf, req.ts, req.moves, req.tf_color, req.swap, req.top_k)
