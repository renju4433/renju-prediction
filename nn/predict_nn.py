"""Inference interface for RenjuDualNet (Policy Head + Value Head) with zero future leakage."""

from __future__ import annotations

import argparse
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
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nn.features import (
    build_input_planes,
    canonicalize_d4,
    coord_to_str,
    invert_d4_coord,
    parse_coord,
    tf_current_color_at_move,
)
from nn.model import RenjuDualNet

DEFAULT_NN_CKPT = ROOT / "artifacts" / "renju_dualnet.pt"
DEFAULT_HISTORY_BUNDLE = ROOT / "artifacts" / "vercel_bundle.json.gz"


class RenjuDualPredictor:
    def __init__(
        self,
        ckpt_path: Union[str, Path] = DEFAULT_NN_CKPT,
        history_bundle_path: Union[str, Path] = DEFAULT_HISTORY_BUNDLE,
    ) -> None:
        with gzip.open(history_bundle_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        self.rules: Dict[int, str] = {int(k): v for k, v in data["rules"].items()}
        self._rule_by_name: Dict[str, int] = {v.strip().lower(): k for k, v in self.rules.items()}
        self.players: Dict[int, dict] = {int(k): v for k, v in data["players"].items()}
        self._players_by_disp: Dict[str, int] = {}
        self._players_by_name: Dict[str, List[int]] = {}
        for pid, info in self.players.items():
            if info.get("d"):
                self._players_by_disp[info["d"].strip()] = pid
            for cand in {
                info.get("n", "").strip().lower(),
                info.get("rn", "").strip().lower(),
                info.get("nn", "").strip().lower(),
            }:
                if cand:
                    self._players_by_name.setdefault(cand, []).append(pid)
        self.history = data["history"]

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.net = RenjuDualNet(
            channels=ckpt.get("channels", 96),
            num_blocks=ckpt.get("num_blocks", 8),
            player_vocab_size=ckpt.get("player_vocab_size", 1),
            player_emb_dim=ckpt.get("player_emb_dim", 16),
        )
        self.net.load_state_dict(ckpt["model_state"])
        self.net.eval()
        self.nn_metrics = ckpt.get("metrics", {})

        vocab_path = ROOT / "artifacts" / "player_vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.player_vocab: Dict[int, int] = {int(k): int(v) for k, v in json.load(f).items()}
        else:
            self.player_vocab = {}

    @staticmethod
    def _normalize_timestamp(ts: str) -> str:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", str(ts).strip())
        if not m:
            raise ValueError(f"Invalid timestamp '{ts}'. Expected 'YYYY-MM-DD'.")
        return m.group(1)

    def _resolve_rule(self, rule: Union[int, str]) -> int:
        if isinstance(rule, int) or (isinstance(rule, str) and rule.strip().isdigit()):
            rid = int(rule)
            if rid in self.rules:
                return rid
            raise ValueError(f"Unknown rule ID: {rid}")
        key = str(rule).strip().lower()
        if key in self._rule_by_name:
            return self._rule_by_name[key]
        matches = [rid for name, rid in self._rule_by_name.items() if key in name]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown or ambiguous rule '{rule}'.")

    def _resolve_player(self, player: Union[int, str]) -> int:
        if isinstance(player, int) or (isinstance(player, str) and player.strip().isdigit()):
            pid = int(player)
            if pid in self.players:
                return pid
            s_pid = str(player).strip()
            if s_pid in self._players_by_disp:
                return self._players_by_disp[s_pid]
            raise ValueError(f"Unknown player ID: {player}")
        key = str(player).strip().lower()
        ids = self._players_by_name.get(key)
        if not ids:
            raise ValueError(f"Unknown player '{player}'.")
        if len(ids) == 1:
            return ids[0]
        opts = ", ".join(f"{pid} ({self.players[pid]['n']})" for pid in ids[:10])
        raise ValueError(f"Multiple players match '{player}': {opts}. Pass player ID.")

    def _get_rating_and_rd(self, pid: int, ts: str) -> Tuple[float, float]:
        hist = self.history
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

    def _get_h2h(self, tf_id: int, ts_id: int, ts: str) -> Tuple[int, float, float]:
        hist = self.history
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

    @torch.no_grad()
    def predict(
        self,
        timestamp: str,
        rule: Union[int, str],
        tentative_first: Union[int, str],
        tentative_second: Union[int, str],
        moves: Optional[List[str]] = None,
        tf_current_color: Optional[Union[int, str]] = None,
        swap: str = "",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Predict Policy (top_k moves) and Value (tf_win, draw, ts_win) without future leakage."""
        date_str = self._normalize_timestamp(timestamp)
        rule_id = self._resolve_rule(rule)
        tf_id = self._resolve_player(tentative_first)
        ts_id = self._resolve_player(tentative_second)

        tf_r, tf_rd = self._get_rating_and_rd(tf_id, date_str)
        ts_r, ts_rd = self._get_rating_and_rd(ts_id, date_str)
        h2h_g, h2h_sr, h2h_dr = self._get_h2h(tf_id, ts_id, date_str)

        moves = moves or []
        board = np.zeros((15, 15), dtype=np.int8)
        parsed: List[Tuple[int, int]] = []
        for idx, m in enumerate(moves):
            xy = parse_coord(m)
            if xy is None:
                raise ValueError(f"Invalid move coordinate: '{m}'")
            x, y = xy
            if board[y, x] != 0:
                raise ValueError(f"Duplicate stone at '{m}'")
            board[y, x] = 1 if (idx % 2 == 0) else 2
            parsed.append((x, y))

        move_idx = len(parsed)
        next_color = 1 if (move_idx % 2 == 0) else 2

        if tf_current_color is not None:
            if isinstance(tf_current_color, str):
                c_str = tf_current_color.strip().lower()
                if c_str in ("1", "b", "black", "黑"):
                    tf_cur = 1
                elif c_str in ("2", "w", "white", "白"):
                    tf_cur = 2
                else:
                    raise ValueError(f"Invalid tf_current_color: '{tf_current_color}'")
            else:
                tf_cur = int(tf_current_color)
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

        tf_vid = self.player_vocab.get(tf_id, 0)
        ts_vid = self.player_vocab.get(ts_id, 0)
        if next_color == tf_cur:
            mover_vid, opp_vid = tf_vid, ts_vid
        else:
            mover_vid, opp_vid = ts_vid, tf_vid

        x_t = torch.from_numpy(planes).unsqueeze(0)
        m_t = torch.tensor([mover_vid], dtype=torch.long)
        o_t = torch.tensor([opp_vid], dtype=torch.long)
        legal_t = torch.from_numpy(legal).unsqueeze(0)
        p_logits, v_logits = self.net(x_t, m_t, o_t)
        p_logits = p_logits.masked_fill(~legal_t, -1e9)
        p_probs = F.softmax(p_logits, dim=1)[0].numpy()
        v_probs = F.softmax(v_logits, dim=1)[0].numpy()

        top_indices = np.argsort(-p_probs)[:top_k]
        top_moves = []
        for idx in top_indices:
            if legal[idx]:
                ox, oy = invert_d4_coord(int(idx % 15), int(idx // 15), csym)
                top_moves.append(
                    {
                        "move": coord_to_str(ox, oy),
                        "prob": round(float(p_probs[idx]), 6),
                    }
                )

        return {
            "timestamp": date_str,
            "rule": {"id": rule_id, "name": self.rules[rule_id]},
            "move_count": move_idx,
            "next_to_move": "black" if next_color == 1 else "white",
            "tf_current_color": "black" if tf_cur == 1 else "white",
            "tentative_first": {
                "id": tf_id,
                "name": self.players[tf_id]["n"],
                "rating": round(tf_r, 2),
                "rd": round(tf_rd, 2),
            },
            "tentative_second": {
                "id": ts_id,
                "name": self.players[ts_id]["n"],
                "rating": round(ts_r, 2),
                "rd": round(ts_rd, 2),
            },
            "policy_top_moves": top_moves,
            "value": {
                "tf_win": round(float(v_probs[0]), 6),
                "draw": round(float(v_probs[1]), 6),
                "ts_win": round(float(v_probs[2]), 6),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Policy + Value with RenjuDualNet.")
    parser.add_argument("--time", required=True, help="Timestamp YYYY-MM-DD")
    parser.add_argument("--rule", required=True, help="Rule ID or name")
    parser.add_argument("--tf", required=True, help="Tentative first player ID or name")
    parser.add_argument("--ts", required=True, help="Tentative second player ID or name")
    parser.add_argument("--moves", default="", help="Space-separated moves, e.g. 'h8 i9 i8'")
    parser.add_argument("--tf-color", default=None, help="Current color of 假先: black or white")
    parser.add_argument("--swap", default="", help="Optional swap sequence if --tf-color omitted")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top policy moves to show")
    args = parser.parse_args()

    moves = args.moves.split() if args.moves.strip() else []
    pred = RenjuDualPredictor()
    res = pred.predict(
        timestamp=args.time,
        rule=args.rule,
        tentative_first=args.tf,
        tentative_second=args.ts,
        moves=moves,
        tf_current_color=args.tf_color,
        swap=args.swap,
        top_k=args.top_k,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
