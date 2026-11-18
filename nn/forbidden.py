"""Renju forbidden-move checker: uses KataGomo C++ DLL on Windows, pure-Python fallback on Linux/Vercel."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
import numpy as np

_DLL_PATH = Path(__file__).resolve().parent / "forbidden.dll"
_LIB = None

if sys.platform == "win32" and _DLL_PATH.exists():
    try:
        _LIB = ctypes.CDLL(str(_DLL_PATH))
        _LIB.kg_is_forbidden.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]
        _LIB.kg_is_forbidden.restype = ctypes.c_int
        _LIB.kg_get_forbidden_map.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ]
        _LIB.kg_get_forbidden_map.restype = None
    except OSError:
        _LIB = None

_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _line_len(board: np.ndarray, x: int, y: int, dx: int, dy: int, color: int) -> int:
    cnt = 1
    cx, cy = x + dx, y + dy
    while 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == color:
        cnt += 1
        cx += dx
        cy += dy
    cx, cy = x - dx, y - dy
    while 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == color:
        cnt += 1
        cx -= dx
        cy -= dy
    return cnt


def _is_five(board: np.ndarray, x: int, y: int, color: int) -> bool:
    for dx, dy in _DIRS:
        if _line_len(board, x, y, dx, dy, color) == 5:
            return True
    return False


def _is_overline(board: np.ndarray, x: int, y: int) -> bool:
    for dx, dy in _DIRS:
        if _line_len(board, x, y, dx, dy, 1) >= 6:
            return True
    return False


def _count_fours_in_dir(board: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
    """Count distinct 4s formed at (x, y) in direction (dx, dy) by checking empty extensions making exact 5."""
    five_pts = 0
    for sign in (1, -1):
        cx, cy = x + sign * dx, y + sign * dy
        while 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == 1:
            cx += sign * dx
            cy += sign * dy
        if 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == 0:
            if _line_len(board, cx, cy, dx, dy, 1) == 5:
                five_pts += 1
                if five_pts == 2:
                    # Check if this is a single straight four (4 consecutive stones) -> counts as 1 four
                    if _line_len(board, x, y, dx, dy, 1) == 4:
                        return 1
                    return 2
    return 1 if five_pts == 1 else 0


def _is_open_three_in_dir(board: np.ndarray, x: int, y: int, dx: int, dy: int) -> bool:
    """Check if placing Black at (x, y) creates an open three in (dx, dy) (can form straight four)."""
    for sign in (1, -1):
        cx, cy = x + sign * dx, y + sign * dy
        while 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == 1:
            cx += sign * dx
            cy += sign * dy
        if 0 <= cx < 15 and 0 <= cy < 15 and board[cy, cx] == 0:
            board[cy, cx] = 1
            # Straight four: exact 4 in a row with both ends open and not forbidden
            is_four = _line_len(board, cx, cy, dx, dy, 1) == 4
            if is_four:
                # Verify both ends of the 4-in-a-row are empty and make exact 5
                open_ends = 0
                for s2 in (1, -1):
                    ex, ey = cx + s2 * dx, cy + s2 * dy
                    while 0 <= ex < 15 and 0 <= ey < 15 and board[ey, ex] == 1:
                        ex += s2 * dx
                        ey += s2 * dy
                    if 0 <= ex < 15 and 0 <= ey < 15 and board[ey, ex] == 0:
                        if _line_len(board, ex, ey, dx, dy, 1) == 5:
                            open_ends += 1
                if open_ends == 2 and not _py_is_forbidden(board, cx, cy):
                    board[cy, cx] = 0
                    return True
            board[cy, cx] = 0
    return False


def _py_is_forbidden(board: np.ndarray, x: int, y: int) -> bool:
    if board[y, x] != 0:
        return False
    board[y, x] = 1
    try:
        if _is_five(board, x, y, 1):
            return False
        if _is_overline(board, x, y):
            return True
        fours = sum(_count_fours_in_dir(board, x, y, dx, dy) for dx, dy in _DIRS)
        if fours >= 2:
            return True
        threes = sum(1 for dx, dy in _DIRS if _is_open_three_in_dir(board, x, y, dx, dy))
        if threes >= 2:
            return True
        return False
    finally:
        board[y, x] = 0


def is_forbidden(board: np.ndarray, x: int, y: int) -> bool:
    """Return True if Black playing at (x, y) is a Renju forbidden move."""
    b = np.ascontiguousarray(board, dtype=np.int8)
    if _LIB is not None:
        return bool(_LIB.kg_is_forbidden(b, int(x), int(y)))
    return _py_is_forbidden(b.copy(), int(x), int(y))


def get_forbidden_map(board: np.ndarray) -> np.ndarray:
    """Return a (15, 15) uint8 mask where 1 indicates a Black forbidden move."""
    b = np.ascontiguousarray(board, dtype=np.int8)
    if _LIB is not None:
        out = np.zeros((15, 15), dtype=np.uint8)
        _LIB.kg_get_forbidden_map(b, out)
        return out
    out = np.zeros((15, 15), dtype=np.uint8)
    bc = b.copy()
    for py in range(15):
        for px in range(15):
            if bc[py, px] == 0 and _py_is_forbidden(bc, px, py):
                out[py, px] = 1
    return out
