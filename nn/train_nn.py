"""Train RenjuDualNet (Policy + Value) on RIF games with KataGomo forbidden planes & causal player features."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxml import etree
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import DEFAULT_RIF_PATH, load_rif
from nn.features import (
    build_input_planes,
    parse_coord,
    tf_current_color_at_move,
)
from nn.model import RenjuDualNet
from rating_engine import compute_causal_glicko


def result_to_label(tf_result: float) -> int:
    if tf_result == 1.0:
        return 0  # 假先胜
    if tf_result == 0.5:
        return 1  # 和棋
    return 2      # 假后胜


def extract_game_moves(rif_path: str = DEFAULT_RIF_PATH) -> Dict[int, List[Tuple[int, int]]]:
    """Parse <move> sequences for all games in RIF XML."""
    game_moves: Dict[int, List[Tuple[int, int]]] = {}
    for _, elem in etree.iterparse(rif_path, events=("end",)):
        if elem.tag == "game":
            gid = int(elem.attrib["id"])
            m_elem = elem.find("move")
            if m_elem is not None and m_elem.text:
                coords = []
                valid = True
                for tok in m_elem.text.split():
                    xy = parse_coord(tok)
                    if xy is None:
                        valid = False
                        break
                    coords.append(xy)
                if valid and len(coords) >= 5:
                    game_moves[gid] = coords
            elem.clear()
    return game_moves


class RenjuPositionDataset(Dataset):
    """Memory-efficient dataset storing compact board states + causal player scalars."""

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        (
            board,
            next_color,
            tf_cur_color,
            rule_id,
            move_idx,
            last_move,
            prev2_move,
            tf_r,
            ts_r,
            tf_rd,
            ts_rd,
            h2h_g,
            h2h_sr,
            h2h_dr,
            target_move_idx,
            value_label,
        ) = self.samples[idx]

        planes, legal_mask = build_input_planes(
            board=board,
            next_color=next_color,
            tf_current_color=tf_cur_color,
            rule_id=rule_id,
            move_idx=move_idx,
            last_move=last_move,
            prev2_move=prev2_move,
            tf_rating=tf_r,
            ts_rating=ts_r,
            tf_rd=tf_rd,
            ts_rd=ts_rd,
            h2h_games=h2h_g,
            h2h_score_rate=h2h_sr,
            h2h_draw_rate=h2h_dr,
        )
        return (
            torch.from_numpy(planes),
            torch.from_numpy(legal_mask),
            torch.tensor(target_move_idx, dtype=torch.long),
            torch.tensor(value_label, dtype=torch.long),
        )


def build_position_samples(
    labeled_samples,
    game_moves: Dict[int, List[Tuple[int, int]]],
    max_train_samples: int = 160000,
    max_test_samples: int = 24000,
    cutoff_date: str = "2024-01-01",
    seed: int = 42,
):
    rng = random.Random(seed)
    train_pos = []
    test_pos = []

    # Separate games chronologically
    train_games = [s for s in labeled_samples if s[0].date < cutoff_date and s[0].game_id in game_moves]
    test_games = [s for s in labeled_samples if s[0].date >= cutoff_date and s[0].game_id in game_moves]

    # Sample recent + diverse training games if needed
    rng.shuffle(train_games)
    rng.shuffle(test_games)

    def collect(games_subset, max_total: int):
        out = []
        for g, tf_r, ts_r, tf_rd, ts_rd, h2h_g, h2h_sr, h2h_dr in games_subset:
            if len(out) >= max_total:
                break
            moves = game_moves[g.game_id]
            n_moves = len(moves)
            if n_moves < 6:
                continue

            # Pick 1 opening position (t in 0..4) and up to 4 mid/late positions (t >= 5)
            open_t = rng.randint(0, min(4, n_moves - 1))
            later_pool = list(range(5, n_moves))
            later_ts = rng.sample(later_pool, k=min(4, len(later_pool)))
            chosen_ts = sorted({open_t, *later_ts})

            val_label = result_to_label(g.tf_result)
            board = np.zeros((15, 15), dtype=np.int8)
            chosen_set = set(chosen_ts)
            max_t = chosen_ts[-1]

            for t in range(max_t + 1):
                if t in chosen_set:
                    nx, ny = moves[t]
                    next_color = 1 if (t % 2 == 0) else 2
                    tf_cur = tf_current_color_at_move(g.rule_id, g.swap, t)
                    last_m = moves[t - 1] if t >= 1 else None
                    prev2_m = moves[t - 2] if t >= 2 else None
                    out.append(
                        (
                            board.copy(),
                            next_color,
                            tf_cur,
                            g.rule_id,
                            t,
                            last_m,
                            prev2_m,
                            tf_r,
                            ts_r,
                            tf_rd,
                            ts_rd,
                            h2h_g,
                            h2h_sr,
                            h2h_dr,
                            ny * 15 + nx,
                            val_label,
                        )
                    )
                mx, my = moves[t]
                board[my, mx] = 1 if (t % 2 == 0) else 2
        return out[:max_total]

    train_pos = collect(train_games, max_train_samples)
    test_pos = collect(test_games, max_test_samples)
    return train_pos, test_pos


@torch.no_grad()
def evaluate(model: RenjuDualNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_n = 0
    pol_loss_sum = 0.0
    val_loss_sum = 0.0
    pol_top1 = 0
    pol_top3 = 0
    val_correct = 0
    brier_sum = 0.0

    for x, legal, y_pol, y_val in loader:
        x = x.to(device)
        legal = legal.to(device)
        y_pol = y_pol.to(device)
        y_val = y_val.to(device)

        p_logits, v_logits = model(x)
        legal_eval = legal.clone()
        legal_eval.scatter_(1, y_pol.unsqueeze(1), True)
        p_logits_masked = p_logits.masked_fill(~legal_eval, -1e9)

        p_loss = F.cross_entropy(p_logits_masked, y_pol, reduction="sum")
        v_loss = F.cross_entropy(v_logits, y_val, reduction="sum")

        pol_loss_sum += float(p_loss.item())
        val_loss_sum += float(v_loss.item())

        pred_top1 = p_logits_masked.argmax(dim=1)
        pol_top1 += int((pred_top1 == y_pol).sum().item())
        top3 = p_logits_masked.topk(3, dim=1).indices
        pol_top3 += int((top3 == y_pol.unsqueeze(1)).any(dim=1).sum().item())

        v_probs = F.softmax(v_logits, dim=1)
        val_correct += int((v_probs.argmax(dim=1) == y_val).sum().item())
        y_onehot = F.one_hot(y_val, num_classes=3).float()
        brier_sum += float(((v_probs - y_onehot) ** 2).sum().item())

        total_n += x.size(0)

    return {
        "samples": total_n,
        "policy_ce": pol_loss_sum / total_n,
        "policy_top1": pol_top1 / total_n,
        "policy_top3": pol_top3 / total_n,
        "value_logloss": val_loss_sum / total_n,
        "value_brier": brier_sum / total_n,
        "value_acc": val_correct / total_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RenjuDualNet with checkpointing & resume.")
    parser.add_argument("--epochs", type=int, default=8, help="Total target epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--train-samples", type=int, default=300000, help="Max train positions")
    parser.add_argument("--test-samples", type=int, default=30000, help="Max test positions")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming latest checkpoint")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading RIF database and move sequences...", flush=True)
    rules, players, tournaments, games = load_rif(DEFAULT_RIF_PATH)
    _, labeled = compute_causal_glicko(
        games, initial_rating=1500.0, c_per_day=8.0, min_rd=60.0
    )
    game_moves = extract_game_moves(DEFAULT_RIF_PATH)
    print(f"Loaded {len(game_moves)} games with full moves in {time.time() - t0:.1f}s.", flush=True)

    train_pos, test_pos = build_position_samples(
        labeled, game_moves, max_train_samples=args.train_samples, max_test_samples=args.test_samples
    )
    print(f"Built {len(train_pos)} train positions (<2024) and {len(test_pos)} test positions (>=2024).", flush=True)

    train_ds = RenjuPositionDataset(train_pos)
    test_ds = RenjuPositionDataset(test_pos)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training RenjuDualNet on device={device} ...", flush=True)

    channels, num_blocks = 96, 8
    model = RenjuDualNet(channels=channels, num_blocks=num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    start_epoch = 1
    history = []

    best_ckpt_path = ROOT / "artifacts" / "renju_dualnet.pt"
    latest_ckpt_path = ROOT / "artifacts" / "renju_dualnet_latest.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scheduler.T_max = epochs
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history", []))
        print(f"Resumed from {latest_ckpt_path} at epoch {start_epoch - 1} (best_v_loss={best_val_loss:.4f}).", flush=True)

    for ep in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_p, running_v, n_seen = 0.0, 0.0, 0

        for x, legal, y_pol, y_val in train_loader:
            x = x.to(device)
            legal = legal.to(device)
            y_pol = y_pol.to(device)
            y_val = y_val.to(device)

            optimizer.zero_grad()
            p_logits, v_logits = model(x)
            legal_train = legal.clone()
            legal_train.scatter_(1, y_pol.unsqueeze(1), True)
            p_logits_masked = p_logits.masked_fill(~legal_train, -1e9)
            p_loss = F.cross_entropy(p_logits_masked, y_pol)
            v_loss = F.cross_entropy(v_logits, y_val)
            loss = p_loss + 1.5 * v_loss
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_p += float(p_loss.item()) * bs
            running_v += float(v_loss.item()) * bs
            n_seen += bs

        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": ep,
            "train_p_ce": running_p / n_seen,
            "train_v_loss": running_v / n_seen,
            **metrics,
        }
        history.append(row)
        print(
            f"Epoch {ep}/{epochs} ({time.time() - ep_t0:.1f}s) | "
            f"train_p={row['train_p_ce']:.4f} train_v={row['train_v_loss']:.4f} | "
            f"test_p_ce={metrics['policy_ce']:.4f} top1={metrics['policy_top1']:.4f} top3={metrics['policy_top3']:.4f} | "
            f"test_v_loss={metrics['value_logloss']:.4f} brier={metrics['value_brier']:.4f} acc={metrics['value_acc']:.4f}",
            flush=True,
        )

        if metrics["value_logloss"] < best_val_loss:
            best_val_loss = metrics["value_logloss"]
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "channels": channels,
                    "num_blocks": num_blocks,
                    "metrics": metrics,
                },
                best_ckpt_path,
            )

        # Always save resumable latest checkpoint after every epoch
        torch.save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "channels": channels,
                "num_blocks": num_blocks,
                "metrics": metrics,
                "history": history,
            },
            latest_ckpt_path,
        )

    print(f"Training complete. Best checkpoint: {best_ckpt_path}, Latest resumable checkpoint: {latest_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
