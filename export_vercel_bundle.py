"""Export a lightweight, dependency-free gzipped JSON bundle for Vercel deployment."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PKL_PATH = ROOT / "artifacts" / "model_bundle.pkl"
OUT_PATH = ROOT / "artifacts" / "vercel_bundle.json.gz"


def main() -> None:
    with open(PKL_PATH, "rb") as f:
        bundle = pickle.load(f)

    booster_dump = bundle["model"].booster_.dump_model()
    hist = bundle["history"]

    # Convert tuple keys (p1, p2) -> "p1,p2" for JSON serialization
    h2h_dates = {f"{k[0]},{k[1]}": v for k, v in hist.h2h_dates.items()}
    h2h_cum_pts = {
        f"{k[0]},{k[1]}": [round(x, 2) for x in v] for k, v in hist.h2h_cum_pts.items()
    }
    h2h_cum_draws = {f"{k[0]},{k[1]}": v for k, v in hist.h2h_cum_draws.items()}

    compact_players = {
        str(pid): {
            "d": info.get("disp_id", ""),
            "n": info.get("full_name", ""),
            "rn": f"{info.get('surname', '').strip()} {info.get('name', '').strip()}".strip(),
            "nn": info.get("native_name", ""),
            "c": info.get("country", ""),
        }
        for pid, info in bundle["players"].items()
    }

    payload = {
        "booster": {
            "num_class": booster_dump["num_class"],
            "tree_info": booster_dump["tree_info"],
            "feature_names": booster_dump["feature_names"],
        },
        "rule_categories": [int(x) for x in bundle["rule_categories"]],
        "rules": {str(k): v for k, v in bundle["rules"].items()},
        "players": compact_players,
        "history": {
            "initial_rating": hist.initial_rating,
            "initial_rd": hist.initial_rd,
            "c_per_day": hist.c_per_day,
            "min_rd": hist.min_rd,
            "max_rd": hist.max_rd,
            "dates": {str(k): v for k, v in hist.dates.items()},
            "ratings": {str(k): [round(x, 2) for x in v] for k, v in hist.ratings.items()},
            "rds": {str(k): [round(x, 2) for x in v] for k, v in hist.rds.items()},
            "h2h_dates": h2h_dates,
            "h2h_cum_pts": h2h_cum_pts,
            "h2h_cum_draws": h2h_cum_draws,
        },
        "metrics": bundle["metrics"],
    }

    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
