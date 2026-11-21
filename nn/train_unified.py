"""Train the unified pre-match + in-game Renju model (policy incl. swap, value).

Usage (GPU, Python 3.12):
    py -3.12 nn/train_unified.py --epochs 4 --train-samples 260000

One model serves both jobs:
  * empty board  -> pre-match win/draw/loss + first move
  * any position -> in-game win/draw/loss + next move (or swap)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lxml import etree
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import DEFAULT_RIF_PATH, load_rif
from nn.features import (
    SWAP_ACTION,
    build_input_planes,
    canonicalize_d4,
    parse_coord,
    swap_decision_plies,
    tf_current_color_at_move,
)
from nn.model import RenjuUnifiedNet
from rating_engine import compute_causal_glicko


def result_to_label(tf_result: float) -> int:
    if tf_result == 1.0:
        return 0
    if tf_result == 0.5:
        return 1
    return 2


def extract_game_moves(rif_path: str = DEFAULT_RIF_PATH) -> Dict[int, List[Tuple[int, int]]]:
    """Parse <move> sequences for every game in the RIF XML."""
    game_moves: Dict[int, List[Tuple[int, int]]] = {}
    for _, elem in etree.iterparse(rif_path, events=("end",)):
        if elem.tag == "game":
            gid = int(elem.attrib["id"])
            m_elem = elem.find("move")
            if m_elem is not None and m_elem.text:
                coords = []
                for tok in m_elem.text.split():
                    xy = parse_coord(tok)
                    if xy is None:
                        coords = []
                        break
                    coords.append(xy)
                if len(coords) >= 6:
                    game_moves[gid] = coords
            elem.clear()
    return game_moves


class PositionDataset(Dataset):
    """Compact samples; boards are expanded to planes lazily, with D4 canonicalisation."""

    def __init__(self, samples: list, id_dropout: float = 0.0) -> None:
        self.samples = samples
        self.id_dropout = id_dropout

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        (
            board,
            next_color,
            tf_color,
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
            tf_vid,
            ts_vid,
            target,
            value_label,
        ) = self.samples[idx]

        board = board.copy()
        if target != SWAP_ACTION:
            tx, ty = target % 15, target // 15
            board, last_move, prev2_move, t_s, _, _ = canonicalize_d4(
                board, last_move, prev2_move, (tx, ty)
            )
            target = t_s[1] * 15 + t_s[0]

        if self.id_dropout > 0.0:
            if random.random() < self.id_dropout:
                tf_vid = 0
            if random.random() < self.id_dropout:
                ts_vid = 0

        planes, legal = build_input_planes(
            board=board,
            next_color=next_color,
            tf_current_color=tf_color,
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
            torch.from_numpy(legal),
            torch.tensor(tf_vid, dtype=torch.long),
            torch.tensor(ts_vid, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(value_label, dtype=torch.long),
            torch.tensor(move_idx == 0, dtype=torch.bool),
        )


def build_player_vocab(games, min_games: int = 20) -> Dict[int, int]:
    cnt: Counter = Counter()
    for g, *_ in games:
        cnt[g.tf_id] += 1
        cnt[g.ts_id] += 1
    vocab: Dict[int, int] = {}
    for pid, c in sorted(cnt.items()):
        if c >= min_games:
            vocab[pid] = len(vocab) + 1
    return vocab


def build_samples(
    labeled,
    game_moves: Dict[int, List[Tuple[int, int]]],
    vocab: Dict[int, int],
    max_samples: int,
    plies_per_game: int = 4,
    seed: int = 42,
) -> list:
    """Sample positions per game; ply 0 (empty board = pre-match) is always included."""
    rng = random.Random(seed)
    out: list = []
    for g, tf_r, ts_r, tf_rd, ts_rd, h2h_g, h2h_sr, h2h_dr in labeled:
        if len(out) >= max_samples:
            break
        moves = game_moves.get(g.game_id)
        if not moves:
            continue
        n_moves = len(moves)
        val_label = result_to_label(g.tf_result)
        tf_vid = vocab.get(g.tf_id, 0)
        ts_vid = vocab.get(g.ts_id, 0)

        # swap decisions: ply -> True when the recorded decision was "swap"
        swap_at: Dict[int, bool] = {}
        for k, ply in enumerate(swap_decision_plies(g.rule_id, g.swap)):
            if k < len(g.swap):
                swap_at[ply] = g.swap[k] == "R"

        plies = {0}
        pool = [t for t in range(1, min(n_moves, 80)) if t not in swap_at]
        rng.shuffle(pool)
        plies.update(pool[: max(0, plies_per_game - 1)])
        for ply in swap_at:
            if 1 <= ply < n_moves:
                plies.add(ply)

        board = np.zeros((15, 15), dtype=np.int8)
        max_t = max(plies)
        for t in range(max_t + 1):
            if t in plies:
                next_color = 1 if (t % 2 == 0) else 2
                tf_color = tf_current_color_at_move(g.rule_id, g.swap, t)
                if swap_at.get(t):
                    target = SWAP_ACTION
                else:
                    mx, my = moves[t]
                    target = my * 15 + mx
                out.append(
                    (
                        board.copy(),
                        next_color,
                        tf_color,
                        g.rule_id,
                        t,
                        moves[t - 1] if t >= 1 else None,
                        moves[t - 2] if t >= 2 else None,
                        tf_r,
                        ts_r,
                        tf_rd,
                        ts_rd,
                        h2h_g,
                        h2h_sr,
                        h2h_dr,
                        tf_vid,
                        ts_vid,
                        target,
                        val_label,
                    )
                )
            if t < n_moves:
                mx, my = moves[t]
                board[my, mx] = 1 if (t % 2 == 0) else 2
    return out[:max_samples]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n = 0
    p_loss_sum = v_loss_sum = 0.0
    top1 = top3 = v_correct = 0
    brier = 0.0
    open_n = open_ok = 0
    open_brier = open_ll = 0.0
    swap_n = swap_ok = 0

    for x, legal, tf_id, ts_id, y_pol, y_val, is_open in loader:
        x, legal = x.to(device), legal.to(device)
        tf_id, ts_id = tf_id.to(device), ts_id.to(device)
        y_pol, y_val, is_open = y_pol.to(device), y_val.to(device), is_open.to(device)

        p_logits, v_logits = model(x, tf_id, ts_id)
        legal_eval = legal.clone()
        legal_eval.scatter_(1, y_pol.unsqueeze(1), True)
        p_masked = p_logits.masked_fill(~legal_eval, -1e9)

        p_loss_sum += float(F.cross_entropy(p_masked, y_pol, reduction="sum").item())
        v_loss_sum += float(F.cross_entropy(v_logits, y_val, reduction="sum").item())

        pred = p_masked.argmax(dim=1)
        top1 += int((pred == y_pol).sum().item())
        top3 += int((p_masked.topk(3, dim=1).indices == y_pol.unsqueeze(1)).any(dim=1).sum().item())

        v_probs = F.softmax(v_logits, dim=1)
        v_correct += int((v_probs.argmax(dim=1) == y_val).sum().item())
        brier += float(((v_probs - F.one_hot(y_val, 3).float()) ** 2).sum().item())

        if is_open.any():
            ov, oy, ol = v_probs[is_open], y_val[is_open], v_logits[is_open]
            open_ll += float(F.cross_entropy(ol, oy, reduction="sum").item())
            open_brier += float(((ov - F.one_hot(oy, 3).float()) ** 2).sum().item())
            open_ok += int((ov.argmax(dim=1) == oy).sum().item())
            open_n += int(is_open.sum().item())

        swap_mask = y_pol == SWAP_ACTION
        if swap_mask.any():
            swap_n += int(swap_mask.sum().item())
            swap_ok += int((pred[swap_mask] == SWAP_ACTION).sum().item())

        n += x.size(0)

    return {
        "samples": n,
        "policy_ce": p_loss_sum / n,
        "policy_top1": top1 / n,
        "policy_top3": top3 / n,
        "value_logloss": v_loss_sum / n,
        "value_brier": brier / n,
        "value_acc": v_correct / n,
        "swap_samples": swap_n,
        "swap_top1": (swap_ok / swap_n) if swap_n else 0.0,
        "open_samples": open_n,
        "open_logloss": (open_ll / open_n) if open_n else 0.0,
        "open_brier": (open_brier / open_n) if open_n else 0.0,
        "open_acc": (open_ok / open_n) if open_n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified Renju model (policy incl. swap + value).")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-samples", type=int, default=260000)
    parser.add_argument("--test-samples", type=int, default=30000)
    parser.add_argument("--plies-per-game", type=int, default=4)
    parser.add_argument("--cutoff", type=str, default="2025-07-01")
    parser.add_argument("--min-player-games", type=int, default=20)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print("Loading RIF + moves ...", flush=True)
    rules, players, tournaments, games = load_rif(DEFAULT_RIF_PATH)
    _, labeled = compute_causal_glicko(games, initial_rating=1500.0, c_per_day=8.0, min_rd=60.0)
    game_moves = extract_game_moves(DEFAULT_RIF_PATH)
    print(f"{len(games)} renju games, {len(labeled)} labeled, {len(game_moves)} with moves ({time.time()-t0:.1f}s)")

    train_games = [s for s in labeled if s[0].date < args.cutoff]
    test_games = [s for s in labeled if s[0].date >= args.cutoff]
    # Sampling order matters: the sample cap would otherwise be filled by the oldest
    # games (which rarely record swap decisions). Shuffle so every era is covered.
    random.Random(42).shuffle(train_games)
    random.Random(7).shuffle(test_games)
    vocab = build_player_vocab(train_games, min_games=args.min_player_games)
    vocab_size = len(vocab) + 1

    print("Building positions ...", flush=True)
    train_samples = build_samples(train_games, game_moves, vocab, args.train_samples, args.plies_per_game, 42)
    test_samples = build_samples(test_games, game_moves, vocab, args.test_samples, args.plies_per_game, 7)
    print(
        f"train={len(train_samples)} test={len(test_samples)} (cutoff {args.cutoff}), "
        f"player_vocab={vocab_size}",
        flush=True,
    )

    train_loader = DataLoader(
        PositionDataset(train_samples, id_dropout=0.30),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        PositionDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    model = RenjuUnifiedNet(
        channels=args.channels,
        num_blocks=args.blocks,
        player_vocab_size=vocab_size,
        player_emb_dim=16,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    best_path = art / "renju_unified.pt"
    latest_path = art / "renju_unified_latest.pt"
    vocab_path = art / "player_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in vocab.items()}, f, separators=(",", ":"))

    best_val = float("inf")
    history = []
    for ep in range(1, args.epochs + 1):
        ep_t = time.time()
        model.train()
        run_p = run_v = 0.0
        seen = 0
        for x, legal, tf_id, ts_id, y_pol, y_val, _ in train_loader:
            x, legal = x.to(device, non_blocking=True), legal.to(device, non_blocking=True)
            tf_id, ts_id = tf_id.to(device), ts_id.to(device)
            y_pol, y_val = y_pol.to(device), y_val.to(device)

            optimizer.zero_grad(set_to_none=True)
            p_logits, v_logits = model(x, tf_id, ts_id)
            legal_t = legal.clone()
            legal_t.scatter_(1, y_pol.unsqueeze(1), True)
            p_masked = p_logits.masked_fill(~legal_t, -1e9)
            loss = F.cross_entropy(p_masked, y_pol) + 1.5 * F.cross_entropy(
                v_logits, y_val, label_smoothing=0.05
            )
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            run_p += float(F.cross_entropy(p_masked, y_pol).item()) * bs
            run_v += float(F.cross_entropy(v_logits, y_val).item()) * bs
            seen += bs

        scheduler.step()
        m = evaluate(model, test_loader, device)
        row = {"epoch": ep, "train_p": run_p / seen, "train_v": run_v / seen, **m}
        history.append(row)
        print(
            f"Epoch {ep}/{args.epochs} ({time.time()-ep_t:.0f}s, lr={scheduler.get_last_lr()[0]:.2e}) | "
            f"train_p={row['train_p']:.4f} train_v={row['train_v']:.4f} | "
            f"test_p_ce={m['policy_ce']:.4f} top1={m['policy_top1']:.4f} top3={m['policy_top3']:.4f} | "
            f"test_v_ll={m['value_logloss']:.4f} brier={m['value_brier']:.4f} acc={m['value_acc']:.4f} | "
            f"open_v_ll={m['open_logloss']:.4f} open_brier={m['open_brier']:.4f} open_acc={m['open_acc']:.4f} | "
            f"swap={m['swap_samples']} swap_top1={m['swap_top1']:.3f}",
            flush=True,
        )

        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "channels": args.channels,
            "num_blocks": args.blocks,
            "player_vocab_size": vocab_size,
            "player_emb_dim": 16,
            "metrics": m,
            "history": history,
        }
        torch.save(ckpt, latest_path)
        if m["value_logloss"] < best_val:
            best_val = m["value_logloss"]
            torch.save(ckpt, best_path)

    print(f"Done. best={best_path}", flush=True)


if __name__ == "__main__":
    main()
