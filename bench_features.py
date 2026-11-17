"""Ablation benchmark: Glicko-1 RD and Head-to-Head (H2H) history features."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from data_loader import load_rif
from train import multiclass_brier_score, result_to_label


def _g_fn(rd: float, q2_3_over_pi2: float) -> float:
    return 1.0 / math.sqrt(1.0 + q2_3_over_pi2 * (rd ** 2))


def main() -> None:
    _, _, _, games = load_rif()
    game_days = [date.fromisoformat(g.date).toordinal() for g in games]

    q = math.log(10.0) / 400.0
    q2_3_over_pi2 = 3.0 * (q ** 2) / (math.pi ** 2)
    c_per_day, min_rd, max_rd, init_rd = 12.0, 60.0, 350.0, 350.0
    c2 = c_per_day ** 2

    r_map, rd_map, last_day = {}, {}, {}
    h2h_games = defaultdict(int)
    h2h_pts = defaultdict(float)
    h2h_draws = defaultdict(int)

    rows = []
    for g, day in zip(games, game_days):
        b, w = g.black_id, g.white_id
        rb, rw = r_map.get(b, 1500.0), r_map.get(w, 1500.0)

        rdb = rd_map.get(b, init_rd)
        if b in last_day and day > last_day[b]:
            rdb = min(math.sqrt(rdb * rdb + c2 * (day - last_day[b])), max_rd)

        rdw = rd_map.get(w, init_rd)
        if w in last_day and day > last_day[w]:
            rdw = min(math.sqrt(rdw * rdw + c2 * (day - last_day[w])), max_rd)

        if g.swap_known and g.tf_id is not None and g.ts_id is not None:
            tf, ts = g.tf_id, g.ts_id
            tf_r, ts_r = (rb, rw) if tf == b else (rw, rb)
            tf_rd, ts_rd = (rdb, rdw) if tf == b else (rdw, rdb)

            n_h2h = h2h_games[(tf, ts)]
            pts_h2h = h2h_pts[(tf, ts)]
            drw_h2h = h2h_draws[(tf, ts)]
            h2h_rate = (pts_h2h + 1.0) / (n_h2h + 2.0)
            h2h_draw_rate = (drw_h2h + 0.08 * 2.0) / (n_h2h + 2.0)

            rows.append(
                (
                    g.date,
                    g.rule_id,
                    tf_r,
                    ts_r,
                    tf_r - ts_r,
                    tf_rd,
                    ts_rd,
                    n_h2h,
                    h2h_rate,
                    h2h_draw_rate,
                    result_to_label(g.tf_result),
                )
            )

        # Glicko-1 update
        gw, gb = _g_fn(rdw, q2_3_over_pi2), _g_fn(rdb, q2_3_over_pi2)
        eb = 1.0 / (1.0 + 10.0 ** (-gw * (rb - rw) / 400.0))
        ew = 1.0 / (1.0 + 10.0 ** (-gb * (rw - rb) / 400.0))
        inv_rd2_b = 1.0 / (rdb * rdb) + (q * gw) ** 2 * eb * (1.0 - eb)
        inv_rd2_w = 1.0 / (rdw * rdw) + (q * gb) ** 2 * ew * (1.0 - ew)
        r_map[b] = rb + (q / inv_rd2_b) * gw * (g.bresult - eb)
        r_map[w] = rw + (q / inv_rd2_w) * gb * ((1.0 - g.bresult) - ew)
        rd_map[b] = max(math.sqrt(1.0 / inv_rd2_b), min_rd)
        rd_map[w] = max(math.sqrt(1.0 / inv_rd2_w), min_rd)
        last_day[b] = last_day[w] = day

        # Causal H2H update
        h2h_games[(b, w)] += 1
        h2h_games[(w, b)] += 1
        h2h_pts[(b, w)] += g.bresult
        h2h_pts[(w, b)] += 1.0 - g.bresult
        if g.bresult == 0.5:
            h2h_draws[(b, w)] += 1
            h2h_draws[(w, b)] += 1

    cols = [
        "date",
        "rule",
        "tf_rating",
        "ts_rating",
        "rating_diff",
        "tf_rd",
        "ts_rd",
        "h2h_games",
        "h2h_score_rate",
        "h2h_draw_rate",
        "label",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["rule"] = df["rule"].astype("category")

    tr = df[df["date"] < "2022-01-01"]
    va = df[(df["date"] >= "2022-01-01") & (df["date"] < "2024-01-01")]
    tv = df[df["date"] < "2024-01-01"]
    te = df[df["date"] >= "2024-01-01"]

    feature_sets = {
        "Baseline (4 features)": ["rule", "tf_rating", "ts_rating", "rating_diff"],
        "+ Glicko RD (tf_rd, ts_rd)": [
            "rule",
            "tf_rating",
            "ts_rating",
            "rating_diff",
            "tf_rd",
            "ts_rd",
        ],
        "+ H2H (games, score_rate, draw_rate)": [
            "rule",
            "tf_rating",
            "ts_rating",
            "rating_diff",
            "h2h_games",
            "h2h_score_rate",
            "h2h_draw_rate",
        ],
        "+ Glicko RD + H2H (All)": [
            "rule",
            "tf_rating",
            "ts_rating",
            "rating_diff",
            "tf_rd",
            "ts_rd",
            "h2h_games",
            "h2h_score_rate",
            "h2h_draw_rate",
        ],
    }

    for name, fcols in feature_sets.items():
        m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            tr[fcols],
            tr["label"],
            eval_set=[(va[fcols], va["label"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        va_loss = log_loss(va["label"], m.predict_proba(va[fcols]), labels=[0, 1, 2])

        m2 = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=m.best_iteration_,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m2.fit(tv[fcols], tv["label"])
        te_prob = m2.predict_proba(te[fcols])
        te_loss = log_loss(te["label"], te_prob, labels=[0, 1, 2])
        te_brier = multiclass_brier_score(te["label"].to_numpy(), te_prob)
        te_acc = accuracy_score(te["label"], np.argmax(te_prob, axis=1))
        print(
            f"{name:38s} | valid_loss={va_loss:.5f} (iter={m.best_iteration_:3d}) "
            f"| test_loss={te_loss:.5f}, brier={te_brier:.5f}, acc={te_acc:.5f}"
        )


if __name__ == "__main__":
    main()
