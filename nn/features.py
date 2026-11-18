"""Board encoding, rule & first-5-moves swap logic, and 32-plane neural network input builder."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from nn.forbidden import get_forbidden_map

# Top 10 RIF rules by game count + "other" (index 10)
TOP_RULE_IDS = [1, 25, 5, 24, 11, 2, 10, 16, 4, 3]
RULE_TO_CHANNEL = {rid: idx for idx, rid in enumerate(TOP_RULE_IDS)}
OTHER_RULE_CHANNEL = len(TOP_RULE_IDS)  # 10

# Gomoku rules without Renju forbidden moves (category == 2 in RIF)
GOMOKU_RULE_IDS = {7, 11, 15, 28}

NUM_INPUT_PLANES = 32


def parse_coord(coord: str) -> Optional[Tuple[int, int]]:
    """Parse RIF coordinate like 'h8', 'a15', 'o1' into (x, y) in 0..14."""
    s = coord.strip().lower()
    if len(s) < 2 or len(s) > 3:
        return None
    col = ord(s[0]) - ord("a")
    if not (0 <= col < 15):
        return None
    try:
        row = int(s[1:]) - 1
    except ValueError:
        return None
    if not (0 <= row < 15):
        return None
    return col, row


def coord_to_str(x: int, y: int) -> str:
    return f"{chr(ord('a') + x)}{y + 1}"


def apply_d4_coord(x: int, y: int, sym: int) -> Tuple[int, int]:
    """Apply one of 8 D4 dihedral symmetries (0..7) to 15x15 coordinate (x, y)."""
    if sym & 4:
        x, y = y, x
    rot = sym & 3
    if rot == 1:
        return 14 - y, x
    if rot == 2:
        return 14 - x, 14 - y
    if rot == 3:
        return y, 14 - x
    return x, y


def apply_d4_board(board: np.ndarray, sym: int) -> np.ndarray:
    """Apply the same D4 symmetry (0..7) to a (15, 15) board array."""
    b = board.T if (sym & 4) else board
    rot = sym & 3
    if rot == 1:
        return np.rot90(b, k=3)
    if rot == 2:
        return np.rot90(b, k=2)
    if rot == 3:
        return np.rot90(b, k=1)
    return b.copy()


def is_renju_rule(rule_id: int) -> bool:
    return rule_id not in GOMOKU_RULE_IDS


def tf_current_color_at_move(rule_id: int, swap: str, move_idx: int) -> int:
    """Return Tentative First (假先) current stone color (1=Black, 2=White) before move `move_idx` (0-based).

    Handles different rules' swap schedules during the first 5 moves:
    - len(swap) == 0: no swaps -> always Black (1).
    - len(swap) == 1: single swap decision after move 3 (move_idx >= 3).
    - len(swap) == 2:
        - Gomoku-Swap2 (rule 11): swap[0] after move 3, swap[1] after move 5 (move_idx >= 5).
        - Soosyrv family: swap[0] after move 3, swap[1] after move 4 (move_idx >= 4).
    - len(swap) >= 3 (Taraguchi / Taraguchi-7 / Taraguchi-10 / Tarannikov):
        - swap[k] occurs after move k+1 (so before move_idx, min(move_idx, len(swap)) decisions occurred).
    """
    n = len(swap)
    if n == 0:
        swaps = 0
    elif n == 1:
        swaps = 1 if (move_idx >= 3 and swap[0] == "R") else 0
    elif n == 2:
        swaps = 0
        if move_idx >= 3 and swap[0] == "R":
            swaps += 1
        second_thresh = 5 if rule_id == 11 else 4
        if move_idx >= second_thresh and swap[1] == "R":
            swaps += 1
    else:
        k = min(move_idx, n)
        swaps = swap[:k].count("R")

    return 1 if (swaps % 2 == 0) else 2


def build_input_planes(
    board: np.ndarray,
    next_color: int,
    tf_current_color: int,
    rule_id: int,
    move_idx: int,
    last_move: Optional[Tuple[int, int]],
    prev2_move: Optional[Tuple[int, int]],
    tf_rating: float,
    ts_rating: float,
    tf_rd: float,
    ts_rd: float,
    h2h_games: int,
    h2h_score_rate: float,
    h2h_draw_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (32, 15, 15) float32 input planes and (225,) legal move mask."""
    planes = np.zeros((NUM_INPUT_PLANES, 15, 15), dtype=np.float32)

    is_black = board == 1
    is_white = board == 2
    is_empty = board == 0
    renju = is_renju_rule(rule_id)

    planes[0] = is_black
    planes[1] = is_white
    planes[2] = is_empty

    if renju:
        fmap = get_forbidden_map(board)
        planes[3] = fmap
    else:
        fmap = None

    if last_move is not None:
        planes[4, last_move[1], last_move[0]] = 1.0
    if prev2_move is not None:
        planes[5, prev2_move[1], prev2_move[0]] = 1.0

    # Side-to-move, tentative-first current color, opening phase
    planes[6].fill(1.0 if next_color == 1 else 0.0)
    planes[7].fill(1.0 if tf_current_color == 1 else 0.0)
    planes[8].fill(1.0 if next_color == tf_current_color else 0.0)
    planes[9].fill(1.0 if move_idx < 5 else 0.0)
    planes[10].fill(min(move_idx, 5) / 5.0)
    planes[11].fill(min(move_idx, 60) / 60.0)

    # Rule channels
    planes[12].fill(1.0 if renju else 0.0)
    r_ch = RULE_TO_CHANNEL.get(rule_id, OTHER_RULE_CHANNEL)
    planes[13 + r_ch].fill(1.0)

    # Causal player features concatenated into input planes
    planes[24].fill((tf_rating - 1500.0) / 400.0)
    planes[25].fill((ts_rating - 1500.0) / 400.0)
    planes[26].fill((tf_rating - ts_rating) / 400.0)
    planes[27].fill(tf_rd / 350.0)
    planes[28].fill(ts_rd / 350.0)
    planes[29].fill(math.log1p(h2h_games) / 4.0)
    planes[30].fill(h2h_score_rate)
    planes[31].fill(h2h_draw_rate)

    # Legal move mask (for Policy Head): empty and (not forbidden if Black to move in Renju)
    legal = is_empty.copy()
    if renju and next_color == 1 and fmap is not None:
        legal &= (fmap == 0)

    return planes, legal.reshape(-1)
