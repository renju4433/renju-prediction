"""Train and evaluate Gradient Boosting models for Renju tentative-first outcome prediction."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from data_loader import DEFAULT_RIF_PATH, load_rif
from rating_engine import compute_causal_glicko


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_COLS = [
    "rule",
    "tf_rating",
    "ts_rating",
    "rating_diff",
    "tf_rd",
    "ts_rd",
    "h2h_games",
    "h2h_score_rate",
    "h2h_draw_rate",
]
CLASS_NAMES = ["假先胜", "和棋", "假后胜"]  # 0, 1, 2

# Chronological split for hyper-parameter tuning. Both windows sit as late as the
# data allows so that the 2024-2026 regime drives the selection; the final model is
# then refit on *every* labeled game up to the database cutoff (see main()).
DEFAULT_TRAIN_END = "2024-07-01"
DEFAULT_VALID_END = "2025-07-01"
# Older boundary, kept only for the "does training on 2024+ actually help?" ablation.
LEGACY_TRAIN_END = "2024-01-01"


def result_to_label(tf_result: float) -> int:
    if tf_result == 1.0:
        return 0  # 假先胜
    if tf_result == 0.5:
        return 1  # 和棋
    return 2      # 假后胜


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_onehot = np.eye(3)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def build_dataframe(labeled_samples) -> pd.DataFrame:
    rows = []
    for g, tf_r, ts_r, tf_rd, ts_rd, n_h2h, h2h_score, h2h_draw in labeled_samples:
        rows.append(
            {
                "date": g.date,
                "rule": g.rule_id,
                "tf_rating": tf_r,
                "ts_rating": ts_r,
                "rating_diff": tf_r - ts_r,
                "tf_rd": tf_rd,
                "ts_rd": ts_rd,
                "h2h_games": n_h2h,
                "h2h_score_rate": h2h_score,
                "h2h_draw_rate": h2h_draw,
                "label": result_to_label(g.tf_result),
            }
        )
    df = pd.DataFrame(rows)
    df["rule"] = df["rule"].astype("category")
    return df


def make_model(params: Dict, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1, 2])),
        "brier": multiclass_brier_score(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pre-match Renju win-rate model.")
    parser.add_argument("--rif", default=DEFAULT_RIF_PATH, help="RenjuNet RIF snapshot to train on")
    parser.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END,
        help="end of the tuning-train window (exclusive)",
    )
    parser.add_argument(
        "--valid-end",
        default=DEFAULT_VALID_END,
        help="end of the tuning-validation window (exclusive); also the out-of-time holdout boundary",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="skip the legacy (<2024-01-01) ablation fit",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_end = args.train_end
    valid_end = args.valid_end

    print(f"Loading RIF database from {args.rif} ...")
    rules, players, tournaments, games = load_rif(args.rif)
    max_date = max(g.date for g in games)
    print(
        f"Loaded {len(rules)} rules, {len(players)} players, {len(tournaments)} tournaments, "
        f"{len(games)} games (through {max_date})."
    )
    print(
        f"Tuning split: train < {train_end} | valid {train_end} .. {valid_end} | "
        f"holdout >= {valid_end}; final model refits on all labeled games."
    )

    best_c = 12.0
    best_min_rd = 80.0
    best_val_loss = float("inf")
    best_params: Dict = {}
    best_iter = 200

    # Tune Glicko-1 (c_per_day, min_rd) and LightGBM hyperparameters on chronological validation set
    for c_per_day, min_rd in [(8.0, 60.0), (8.0, 80.0), (12.0, 60.0), (12.0, 80.0)]:
        _, labeled = compute_causal_glicko(
            games, initial_rating=1500.0, c_per_day=c_per_day, min_rd=min_rd
        )
        df = build_dataframe(labeled)

        train_df = df[df["date"] < train_end]
        valid_df = df[(df["date"] >= train_end) & (df["date"] < valid_end)]

        X_tr, y_tr = train_df[FEATURE_COLS], train_df["label"].to_numpy()
        X_va, y_va = valid_df[FEATURE_COLS], valid_df["label"].to_numpy()

        for lr, num_leaves, min_child in [(0.03, 31, 50), (0.05, 31, 100)]:
            model = make_model(
                {
                    "learning_rate": lr,
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child,
                },
                n_estimators=1500,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            va_prob = model.predict_proba(X_va)
            va_loss = log_loss(y_va, va_prob, labels=[0, 1, 2])
            print(
                f"Glicko c={c_per_day} min_rd={min_rd} lr={lr} leaves={num_leaves} min_child={min_child} "
                f"-> valid_logloss={va_loss:.5f} (best_iter={model.best_iteration_})"
            )
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_c = c_per_day
                best_min_rd = min_rd
                best_iter = int(model.best_iteration_ or 200)
                best_params = {
                    "learning_rate": lr,
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child,
                }

    print(
        f"\nBest config: Glicko c={best_c}, min_rd={best_min_rd}, "
        f"{best_params}, best_iter={best_iter}, valid_logloss={best_val_loss:.5f}"
    )

    # Recompute causal Glicko with best parameters
    history, labeled = compute_causal_glicko(
        games, initial_rating=1500.0, c_per_day=best_c, min_rd=best_min_rd
    )
    df = build_dataframe(labeled)

    train_val_df = df[df["date"] < valid_end]
    test_df = df[df["date"] >= valid_end]

    X_tv, y_tv = train_val_df[FEATURE_COLS], train_val_df["label"].to_numpy()
    X_te, y_te = test_df[FEATURE_COLS], test_df["label"].to_numpy()

    print(
        f"\nSamples: all={len(df)} (through {df['date'].max()}), "
        f"train+valid={len(train_val_df)}, holdout={len(test_df)}"
    )

    # Strictly out-of-time evaluation: everything before the holdout trains, holdout tests
    eval_model = make_model(best_params, best_iter)
    eval_model.fit(X_tv, y_tv)
    te_prob = eval_model.predict_proba(X_te)
    te_pred = np.argmax(te_prob, axis=1)
    te = evaluate(y_te, te_prob, te_pred)

    print(f"\n=== Out-of-time test (date >= {valid_end}) ===")
    print(f"Model trained on dates < {valid_end}: {len(train_val_df)} samples")
    print(f"Holdout samples (>= {valid_end}): {len(test_df)}")
    print(f"Test Log-Loss    : {te['logloss']:.5f}")
    print(f"Test Brier Score : {te['brier']:.5f}")
    print(f"Test Accuracy    : {te['accuracy']:.5f}")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        actual_rate = float(np.mean(y_te == c_idx))
        pred_rate = float(np.mean(te_prob[:, c_idx]))
        print(f"  Class {c_idx} ({c_name}): actual={actual_rate:.4f}, mean_pred={pred_rate:.4f}")

    # Ablation: same hyper-parameters, but a strictly older training window
    # (nothing after 2024-01-01). Tells us whether the 2024+ games actually add signal.
    ablation: Dict[str, object] = {}
    if not args.no_ablation and LEGACY_TRAIN_END < valid_end:
        legacy_train = df[df["date"] < LEGACY_TRAIN_END]
        legacy_model = make_model(best_params, best_iter)
        legacy_model.fit(legacy_train[FEATURE_COLS], legacy_train["label"].to_numpy())
        lg_prob = legacy_model.predict_proba(X_te)
        lg = evaluate(y_te, lg_prob, np.argmax(lg_prob, axis=1))
        ablation = {
            "train_end": LEGACY_TRAIN_END,
            "train_count": len(legacy_train),
            "test_logloss": lg["logloss"],
            "test_brier": lg["brier"],
            "test_accuracy": lg["accuracy"],
        }
        print(
            f"\n=== Ablation on the same holdout ===\n"
            f"Trained < {LEGACY_TRAIN_END} ({len(legacy_train)} samples): "
            f"logloss={lg['logloss']:.5f}, brier={lg['brier']:.5f}, acc={lg['accuracy']:.5f}\n"
            f"Trained < {valid_end} ({len(train_val_df)} samples): "
            f"logloss={te['logloss']:.5f}, brier={te['brier']:.5f}, acc={te['accuracy']:.5f}"
        )

    # Refit the shipped model on every labeled game of the snapshot, using the
    # validated hyper-parameters. This is the "all-period" model served by predict/API.
    X_all, y_all = df[FEATURE_COLS], df["label"].to_numpy()
    final_model = make_model(best_params, best_iter)
    final_model.fit(X_all, y_all)
    print(
        f"\nFinal model refit on all labeled data: {len(df)} samples "
        f"({df['date'].min()} .. {df['date'].max()})."
    )

    # Save bundle for point-in-time inference
    bundle = {
        "model": final_model,
        "eval_model": eval_model,
        "history": history,
        "rules": rules,
        "players": players,
        "rule_categories": list(df["rule"].cat.categories),
        "feature_cols": FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "metrics": {
            "rating_system": "glicko-1",
            "rif_snapshot": Path(args.rif).name,
            "best_c_per_day": best_c,
            "best_min_rd": best_min_rd,
            "best_params": best_params,
            "best_iter": best_iter,
            "valid_logloss": best_val_loss,
            "tuning_train_end": train_end,
            "tuning_valid_end": valid_end,
            "test_logloss": te["logloss"],
            "test_brier": te["brier"],
            "test_accuracy": te["accuracy"],
            "train_val_count": len(train_val_df),
            "test_count": len(test_df),
            "total_labeled_count": len(df),
            "data_first_date": str(df["date"].min()),
            "data_last_date": str(df["date"].max()),
            "final_model_trains_on": "all_labeled_games",
            "ablation_legacy": ablation,
        },
    }

    bundle_path = ARTIFACT_DIR / "model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, ensure_ascii=False, indent=2)

    print(f"\nSaved model bundle to {bundle_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
