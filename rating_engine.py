"""Strictly causal online Glicko-1 + Head-to-Head (H2H) engine with point-in-time lookup."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

from data_loader import GameRecord

_Q = math.log(10.0) / 400.0
_Q2_3_OVER_PI2 = 3.0 * (_Q ** 2) / (math.pi ** 2)


def _g_fn(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + _Q2_3_OVER_PI2 * (rd ** 2))


@dataclass
class RatingHistory:
    initial_rating: float
    initial_rd: float
    c_per_day: float
    min_rd: float
    max_rd: float
    # player_id -> sorted_dates, ratings_after_game, rds_after_game
    dates: Dict[int, List[str]]
    ratings: Dict[int, List[float]]
    rds: Dict[int, List[float]]
    # undirected pair (min_id, max_id) -> sorted_dates, cum_min_id_pts, cum_draws
    h2h_dates: Dict[Tuple[int, int], List[str]]
    h2h_cum_pts: Dict[Tuple[int, int], List[float]]
    h2h_cum_draws: Dict[Tuple[int, int], List[int]]

    def get_rating_and_rd_at(self, player_id: int, timestamp: str) -> Tuple[float, float]:
        """Return player's (rating, rd) strictly before `timestamp` (YYYY-MM-DD)."""
        p_dates = self.dates.get(player_id)
        if not p_dates:
            return self.initial_rating, self.initial_rd
        idx = bisect_left(p_dates, timestamp)
        if idx == 0:
            return self.initial_rating, self.initial_rd
        r = self.ratings[player_id][idx - 1]
        rd = self.rds[player_id][idx - 1]
        dt = date.fromisoformat(timestamp).toordinal() - date.fromisoformat(p_dates[idx - 1]).toordinal()
        if dt > 0:
            rd = min(math.sqrt(rd * rd + (self.c_per_day ** 2) * dt), self.max_rd)
        return r, rd

    def get_rating_at(self, player_id: int, timestamp: str) -> float:
        r, _ = self.get_rating_and_rd_at(player_id, timestamp)
        return r

    def get_h2h_at(self, tf_id: int, ts_id: int, timestamp: str) -> Tuple[int, float, float]:
        """Return (h2h_games, h2h_score_rate, h2h_draw_rate) strictly before `timestamp`."""
        key = (tf_id, ts_id) if tf_id < ts_id else (ts_id, tf_id)
        dates_list = self.h2h_dates.get(key)
        if not dates_list:
            return 0, 0.5, 0.08
        idx = bisect_left(dates_list, timestamp)
        if idx == 0:
            return 0, 0.5, 0.08
        n_h2h = idx
        min_pts = self.h2h_cum_pts[key][idx - 1]
        draws = self.h2h_cum_draws[key][idx - 1]
        tf_pts = min_pts if tf_id < ts_id else (n_h2h - min_pts)
        score_rate = (tf_pts + 1.0) / (n_h2h + 2.0)
        draw_rate = (draws + 0.08 * 2.0) / (n_h2h + 2.0)
        return n_h2h, score_rate, draw_rate


def compute_causal_glicko(
    games: List[GameRecord],
    initial_rating: float = 1500.0,
    initial_rd: float = 350.0,
    c_per_day: float = 12.0,
    min_rd: float = 60.0,
    max_rd: float = 350.0,
) -> Tuple[RatingHistory, List[Tuple[GameRecord, float, float, float, float, int, float, float]]]:
    """Compute sequential Glicko-1 ratings + H2H stats in chronological order with zero future leakage."""
    r_map: Dict[int, float] = {}
    rd_map: Dict[int, float] = {}
    last_day: Dict[int, int] = {}

    hist_dates: Dict[int, List[str]] = defaultdict(list)
    hist_ratings: Dict[int, List[float]] = defaultdict(list)
    hist_rds: Dict[int, List[float]] = defaultdict(list)

    h2h_dates: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    h2h_cum_pts: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    h2h_cum_draws: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    labeled_samples = []
    c2 = c_per_day * c_per_day
    day_cache: Dict[str, int] = {}

    for g in games:
        day = day_cache.get(g.date)
        if day is None:
            day = date.fromisoformat(g.date).toordinal()
            day_cache[g.date] = day

        b, w = g.black_id, g.white_id
        rb = r_map.get(b, initial_rating)
        rw = r_map.get(w, initial_rating)

        rdb = rd_map.get(b, initial_rd)
        if b in last_day and day > last_day[b]:
            rdb = min(math.sqrt(rdb * rdb + c2 * (day - last_day[b])), max_rd)

        rdw = rd_map.get(w, initial_rd)
        if w in last_day and day > last_day[w]:
            rdw = min(math.sqrt(rdw * rdw + c2 * (day - last_day[w])), max_rd)

        key = (b, w) if b < w else (w, b)
        prev_dates = h2h_dates[key]
        n_h2h = len(prev_dates)
        prev_min_pts = h2h_cum_pts[key][-1] if n_h2h > 0 else 0.0
        prev_draws = h2h_cum_draws[key][-1] if n_h2h > 0 else 0

        if g.swap_known and g.tf_id is not None and g.ts_id is not None:
            tf, ts = g.tf_id, g.ts_id
            tf_r, ts_r = (rb, rw) if tf == b else (rw, rb)
            tf_rd, ts_rd = (rdb, rdw) if tf == b else (rdw, rdb)
            tf_pts = prev_min_pts if tf < ts else (n_h2h - prev_min_pts)
            h2h_score_rate = (tf_pts + 1.0) / (n_h2h + 2.0)
            h2h_draw_rate = (prev_draws + 0.08 * 2.0) / (n_h2h + 2.0)
            labeled_samples.append(
                (g, tf_r, ts_r, tf_rd, ts_rd, n_h2h, h2h_score_rate, h2h_draw_rate)
            )

        # Glicko-1 update using final colors and bresult
        gw = _g_fn(rdw)
        gb = _g_fn(rdb)
        eb = 1.0 / (1.0 + 10.0 ** (-gw * (rb - rw) / 400.0))
        ew = 1.0 / (1.0 + 10.0 ** (-gb * (rw - rb) / 400.0))

        inv_rd2_b = 1.0 / (rdb * rdb) + (_Q * gw) ** 2 * eb * (1.0 - eb)
        inv_rd2_w = 1.0 / (rdw * rdw) + (_Q * gb) ** 2 * ew * (1.0 - ew)

        new_rb = rb + (_Q / inv_rd2_b) * gw * (g.bresult - eb)
        new_rw = rw + (_Q / inv_rd2_w) * gb * ((1.0 - g.bresult) - ew)
        new_rdb = max(math.sqrt(1.0 / inv_rd2_b), min_rd)
        new_rdw = max(math.sqrt(1.0 / inv_rd2_w), min_rd)

        r_map[b], rd_map[b], last_day[b] = new_rb, new_rdb, day
        r_map[w], rd_map[w], last_day[w] = new_rw, new_rdw, day

        hist_dates[b].append(g.date)
        hist_ratings[b].append(new_rb)
        hist_rds[b].append(new_rdb)
        hist_dates[w].append(g.date)
        hist_ratings[w].append(new_rw)
        hist_rds[w].append(new_rdw)

        # H2H update
        min_add = g.bresult if b < w else (1.0 - g.bresult)
        draw_add = 1 if g.bresult == 0.5 else 0
        h2h_dates[key].append(g.date)
        h2h_cum_pts[key].append(prev_min_pts + min_add)
        h2h_cum_draws[key].append(prev_draws + draw_add)

    history = RatingHistory(
        initial_rating=initial_rating,
        initial_rd=initial_rd,
        c_per_day=c_per_day,
        min_rd=min_rd,
        max_rd=max_rd,
        dates=dict(hist_dates),
        ratings=dict(hist_ratings),
        rds=dict(hist_rds),
        h2h_dates=dict(h2h_dates),
        h2h_cum_pts=dict(h2h_cum_pts),
        h2h_cum_draws=dict(h2h_cum_draws),
    )
    return history, labeled_samples
