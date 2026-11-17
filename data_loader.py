"""Load and parse RenjuNet RIF XML database, including tentative-first/second inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from lxml import etree


DEFAULT_RIF_PATH = r"c:\Users\yaolz\Downloads\renjunet_v10_20261116.rif"


@dataclass(frozen=True)
class GameRecord:
    game_id: int
    tournament_id: int
    date: str  # YYYY-MM-DD (tournament start date)
    round_str: str
    round_key: Tuple[int, int, str]
    rule_id: int
    black_id: int
    white_id: int
    bresult: float  # 1.0 (black win), 0.5 (draw), 0.0 (white win)
    swap: str
    swap_known: bool
    tf_id: Optional[int]  # Tentative First (假先) player ID
    ts_id: Optional[int]  # Tentative Second (假后) player ID
    tf_result: Optional[float]  # 1.0 (假先胜), 0.5 (和棋), 0.0 (假后胜)


def parse_round_key(round_str: str) -> Tuple[int, int, str]:
    """Convert round strings like '1', '2', '10', 'F1', 'A3', '' into a sortable key."""
    s = (round_str or "").strip()
    if not s:
        return (2, 0, "")
    if s.isdigit():
        return (0, int(s), s)
    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    if m:
        return (1, int(m.group(2)), m.group(1))
    return (1, 0, s)


def infer_tentative_players(
    black_id: int, white_id: int, bresult: float, swap: str
) -> Tuple[bool, Optional[int], Optional[int], Optional[float]]:
    """Reverse-engineer tentative first (假先) and tentative second (假后) from final colors and swap sequence.

    In RIF format, each 'R' in `swap` represents a color swap between the two players,
    '-' represents no swap, and 'x' represents an unknown swap decision.
    - Even number of 'R's => tentative_first = final black, tentative_second = final white.
    - Odd number of 'R's  => tentative_first = final white, tentative_second = final black.
    """
    if "x" in swap:
        return False, None, None, None

    n_swaps = swap.count("R")
    if n_swaps % 2 == 0:
        return True, black_id, white_id, bresult
    return True, white_id, black_id, 1.0 - bresult


def load_rif(
    rif_path: str = DEFAULT_RIF_PATH,
) -> Tuple[Dict[int, str], Dict[int, dict], Dict[int, dict], List[GameRecord]]:
    """Parse RIF XML and return (rules, players, tournaments, chronologically_sorted_games)."""
    tree = etree.parse(rif_path)
    root = tree.getroot()

    rules: Dict[int, str] = {}
    for r in root.findall("./rules/rule"):
        rules[int(r.attrib["id"])] = r.attrib["name"]

    players: Dict[int, dict] = {}
    for p in root.findall("./players/player"):
        pid = int(p.attrib["id"])
        name = p.attrib.get("name", "").strip()
        surname = p.attrib.get("surname", "").strip()
        full_name = f"{name} {surname}".strip()
        players[pid] = {
            "id": pid,
            "disp_id": p.attrib.get("disp_id", ""),
            "name": name,
            "surname": surname,
            "full_name": full_name,
            "native_name": p.attrib.get("native_name", "").strip(),
            "country": p.attrib.get("country", ""),
        }

    tournaments: Dict[int, dict] = {}
    for t in root.findall("./tournaments/tournament"):
        tid = int(t.attrib["id"])
        tournaments[tid] = {
            "id": tid,
            "name": t.attrib.get("name", ""),
            "start": t.attrib["start"],
            "end": t.attrib["end"],
            "rated": int(t.attrib.get("rated", "1")),
            "rule": int(t.attrib.get("rule", "0")),
        }

    games: List[GameRecord] = []
    for g in root.findall("./games/game"):
        gid = int(g.attrib["id"])
        tid = int(g.attrib["tournament"])
        t_info = tournaments[tid]
        date_str = t_info["start"]
        round_str = g.attrib.get("round", "")
        rule_id = int(g.attrib["rule"])
        black_id = int(g.attrib["black"])
        white_id = int(g.attrib["white"])
        bresult = float(g.attrib["bresult"])
        swap = g.attrib.get("swap", "")

        swap_known, tf_id, ts_id, tf_result = infer_tentative_players(
            black_id, white_id, bresult, swap
        )
        games.append(
            GameRecord(
                game_id=gid,
                tournament_id=tid,
                date=date_str,
                round_str=round_str,
                round_key=parse_round_key(round_str),
                rule_id=rule_id,
                black_id=black_id,
                white_id=white_id,
                bresult=bresult,
                swap=swap,
                swap_known=swap_known,
                tf_id=tf_id,
                ts_id=ts_id,
                tf_result=tf_result,
            )
        )

    # Strict chronological sort: (date, tournament_id, round_key, game_id)
    games.sort(key=lambda x: (x.date, x.tournament_id, x.round_key, x.game_id))
    return rules, players, tournaments, games
