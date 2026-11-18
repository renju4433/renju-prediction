"""Point-in-time prediction interface for Renju tentative-first/draw/tentative-second."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parent / "artifacts" / "model_bundle.pkl"


class RenjuPredictor:
    """Predict P(假先胜), P(和棋), P(假后胜) at any timestamp without future leakage."""

    def __init__(self, bundle_path: Union[str, Path] = DEFAULT_BUNDLE_PATH) -> None:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.eval_model = bundle["eval_model"]
        self.history = bundle["history"]
        self.rules: Dict[int, str] = bundle["rules"]
        self.players: Dict[int, dict] = bundle["players"]
        self.rule_categories: List[int] = bundle["rule_categories"]
        self.feature_cols: List[str] = bundle["feature_cols"]
        self.class_names: List[str] = bundle["class_names"]
        self.metrics: Dict[str, Any] = bundle["metrics"]

        # Build fast lookup indices for rules and players
        self._rule_by_name: Dict[str, int] = {
            name.strip().lower(): rid for rid, name in self.rules.items()
        }
        self._players_by_disp: Dict[str, int] = {}
        self._players_by_name: Dict[str, List[int]] = {}
        for pid, info in self.players.items():
            if info.get("disp_id"):
                self._players_by_disp[info["disp_id"].strip()] = pid
            candidates = {
                info.get("full_name", "").strip().lower(),
                f"{info.get('surname', '').strip()} {info.get('name', '').strip()}".strip().lower(),
                info.get("native_name", "").strip().lower(),
            }
            for cname in candidates:
                if cname:
                    self._players_by_name.setdefault(cname, []).append(pid)

    @staticmethod
    def normalize_timestamp(ts: str) -> str:
        """Normalize timestamp to YYYY-MM-DD for causal date comparison."""
        s = str(ts).strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
        if not m:
            raise ValueError(
                f"Invalid timestamp '{ts}'. Expected 'YYYY-MM-DD' (or ISO datetime starting with YYYY-MM-DD)."
            )
        return m.group(1)

    def resolve_rule(self, rule: Union[int, str]) -> int:
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
        if len(matches) > 1:
            opts = ", ".join(f"{rid}:{self.rules[rid]}" for rid in matches)
            raise ValueError(f"Ambiguous rule '{rule}'. Matches: {opts}")
        raise ValueError(f"Unknown rule '{rule}'.")

    def resolve_player(self, player: Union[int, str]) -> int:
        if isinstance(player, int) or (isinstance(player, str) and player.strip().isdigit()):
            pid = int(player)
            if pid in self.players:
                return pid
            # Fallback to disp_id if numeric
            s_pid = str(player).strip()
            if s_pid in self._players_by_disp:
                return self._players_by_disp[s_pid]
            raise ValueError(f"Unknown player ID: {player}")

        key = str(player).strip().lower()
        if key in self._players_by_name:
            ids = self._players_by_name[key]
            if len(ids) == 1:
                return ids[0]
            opts = ", ".join(
                f"{pid} ({self.players[pid]['full_name']}, country={self.players[pid]['country']})"
                for pid in ids
            )
            raise ValueError(f"Multiple players match '{player}': {opts}. Please pass player ID.")

        raise ValueError(f"Unknown player '{player}'.")

    def predict(
        self,
        timestamp: str,
        rule: Union[int, str],
        tentative_first: Union[int, str],
        tentative_second: Union[int, str],
        use_eval_model: bool = False,
    ) -> Dict[str, Any]:
        """Predict probabilities [假先胜, 和棋, 假后胜] at `timestamp` with zero future leakage."""
        date_str = self.normalize_timestamp(timestamp)
        rule_id = self.resolve_rule(rule)
        tf_id = self.resolve_player(tentative_first)
        ts_id = self.resolve_player(tentative_second)

        # Strictly causal lookup: only games with game.date < date_str are used
        tf_rating, tf_rd = self.history.get_rating_and_rd_at(tf_id, date_str)
        ts_rating, ts_rd = self.history.get_rating_and_rd_at(ts_id, date_str)
        rating_diff = tf_rating - ts_rating
        h2h_games, h2h_score_rate, h2h_draw_rate = self.history.get_h2h_at(
            tf_id, ts_id, date_str
        )

        X = pd.DataFrame(
            {
                "rule": pd.Categorical([rule_id], categories=self.rule_categories),
                "tf_rating": [tf_rating],
                "ts_rating": [ts_rating],
                "rating_diff": [rating_diff],
                "tf_rd": [tf_rd],
                "ts_rd": [ts_rd],
                "h2h_games": [h2h_games],
                "h2h_score_rate": [h2h_score_rate],
                "h2h_draw_rate": [h2h_draw_rate],
            }
        )[self.feature_cols]

        clf = self.eval_model if use_eval_model else self.model
        probs = clf.predict_proba(X)[0]

        return {
            "timestamp": date_str,
            "rule": {"id": rule_id, "name": self.rules[rule_id]},
            "tentative_first": {
                "id": tf_id,
                "name": self.players[tf_id]["full_name"],
                "rating": round(tf_rating, 2),
                "rd": round(tf_rd, 2),
            },
            "tentative_second": {
                "id": ts_id,
                "name": self.players[ts_id]["full_name"],
                "rating": round(ts_rating, 2),
                "rd": round(ts_rd, 2),
            },
            "features": {
                "rule": rule_id,
                "tf_rating": round(tf_rating, 2),
                "ts_rating": round(ts_rating, 2),
                "rating_diff": round(rating_diff, 2),
                "tf_rd": round(tf_rd, 2),
                "ts_rd": round(ts_rd, 2),
                "h2h_games": h2h_games,
                "h2h_score_rate": round(h2h_score_rate, 4),
                "h2h_draw_rate": round(h2h_draw_rate, 4),
            },
            "probabilities": {
                "tf_win_prob": round(float(probs[0]), 6),
                "draw_prob": round(float(probs[1]), 6),
                "ts_win_prob": round(float(probs[2]), 6),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Renju tentative-first outcome probabilities.")
    parser.add_argument("--time", required=True, help="Timestamp (YYYY-MM-DD)")
    parser.add_argument("--rule", required=True, help="Rule ID or name (e.g. Taraguchi-10, 25)")
    parser.add_argument("--tf", required=True, help="Tentative first (假先) player ID or full name")
    parser.add_argument("--ts", required=True, help="Tentative second (假后) player ID or full name")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH), help="Path to model_bundle.pkl")
    args = parser.parse_args()

    predictor = RenjuPredictor(args.bundle)
    res = predictor.predict(args.time, args.rule, args.tf, args.ts)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
