"""Train and evaluate Gradient Boosting models for Renju tentative-first outcome prediction."""

from __future__ import annotations

import json
import os
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


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading RIF database from {DEFAULT_RIF_PATH} ...")
    rules, players, tournaments, games = load_rif(DEFAULT_RIF_PATH)
    print(f"Loaded {len(rules)} rules, {len(players)} players, {len(tournaments)} tournaments, {len(games)} games.")

    # Chronological split boundaries (zero future leakage)
    train_end = "2022-01-01"
    valid_end = "2024-01-01"

    best_c = 12.0
    best_min_rd = 80.0
    best_val_loss = float("inf")
    best_params = None
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
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=1500,
                learning_rate=lr,
                num_leaves=num_leaves,
                min_child_samples=min_child,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
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

    # Strictly out-of-time evaluation: train on < 2024-01-01, test on >= 2024-01-01
    eval_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=best_iter,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **best_params,
    )
    eval_model.fit(X_tv, y_tv)
    te_prob = eval_model.predict_proba(X_te)
    te_pred = np.argmax(te_prob, axis=1)

    te_loss = log_loss(y_te, te_prob, labels=[0, 1, 2])
    te_brier = multiclass_brier_score(y_te, te_prob)
    te_acc = accuracy_score(y_te, te_pred)

    print("\n=== Strictly Out-of-Time Test Evaluation (date >= 2024-01-01) ===")
    print(f"Train+Valid samples (< 2024-01-01): {len(train_val_df)}")
    print(f"Test samples       (>= 2024-01-01): {len(test_df)}")
    print(f"Test Log-Loss    : {te_loss:.5f}")
    print(f"Test Brier Score : {te_brier:.5f}")
    print(f"Test Accuracy    : {te_acc:.5f}")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        actual_rate = float(np.mean(y_te == c_idx))
        pred_rate = float(np.mean(te_prob[:, c_idx]))
        print(f"  Class {c_idx} ({c_name}): actual={actual_rate:.4f}, mean_pred={pred_rate:.4f}")

    # Refit final model on all data up to database cutoff using the validated hyperparameters
    X_all, y_all = df[FEATURE_COLS], df["label"].to_numpy()
    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=best_iter,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **best_params,
    )
    final_model.fit(X_all, y_all)

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
            "best_c_per_day": best_c,
            "best_min_rd": best_min_rd,
            "best_params": best_params,
            "best_iter": best_iter,
            "valid_logloss": best_val_loss,
            "test_logloss": te_loss,
            "test_brier": te_brier,
            "test_accuracy": te_acc,
            "train_val_count": len(train_val_df),
            "test_count": len(test_df),
            "total_labeled_count": len(df),
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
