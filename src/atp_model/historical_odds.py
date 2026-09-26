"""Historical closing odds (tennis-data.co.uk) for honest market benchmarking.

The file ``data/odds/tennis_data_atp.csv.gz`` holds one row per completed ATP
main-draw match since 2010 with Pinnacle closing odds and the market average.
Source: http://www.tennis-data.co.uk (free for non-commercial research use; check
their terms before commercial redistribution).

These odds are used ONLY after the fact:
  * to measure whether the model is more or less accurate than the market;
  * to fit the market-blend layer (how much weight the model deserves on top of
    the Pinnacle price);
  * to backtest the betting rules shown in the app.
They never enter the pre-match feature vector.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

ODDS_PATH = ROOT / "data" / "odds" / "tennis_data_atp.csv.gz"


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z]+", " ", text.lower()).split())


def td_key(name) -> str:
    """'Auger-Aliassime F.' -> 'auger aliassime|f'."""
    name = str(name or "").strip()
    m = re.match(r"^(.*?)\s+((?:[A-Z]\.?-?)+)\.?$", name)
    if not m:
        return _norm(name)
    return _norm(m.group(1)) + "|" + m.group(2)[0].lower()


def full_name_keys(name) -> set[str]:
    """All plausible 'surname|initial' keys for a full name (multi-part surnames)."""
    tokens = _norm(name).split()
    if len(tokens) < 2:
        return {_norm(name)}
    keys = {" ".join(tokens[k:]) + "|" + tokens[0][0] for k in range(1, len(tokens))}
    keys.add(" ".join(tokens[:-1]) + "|" + tokens[-1][0])
    return keys


def load_odds(path: Path = ODDS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wk"] = df["winner"].map(td_key)
    df["lk"] = df["loser"].map(td_key)
    return df


def attach_odds(frame: pd.DataFrame, odds: pd.DataFrame, *, winner_col="winner_name",
                loser_col="loser_name", date_col="date") -> pd.DataFrame:
    """Add ps_w/ps_l/avg_w/avg_l (winner/loser closing odds) to a match frame."""
    out = frame.copy()
    for c in ("ps_w", "ps_l", "avg_w", "avg_l"):
        out[c] = np.nan
    out["odds_series"] = None
    if odds.empty or out.empty:
        return out
    index: dict[tuple[str, str], list[int]] = {}
    for i, (wk, lk) in enumerate(zip(odds["wk"], odds["lk"])):
        index.setdefault((wk, lk), []).append(i)
    odates = odds["date"].to_numpy()
    cols = {"ps_w": "ps_w", "ps_l": "ps_l", "avg_w": "avg_w", "avg_l": "avg_l"}
    ovals = {k: odds[v].to_numpy() for k, v in cols.items()}
    series = odds["series"].to_numpy() if "series" in odds.columns else None
    hits = np.full(len(out), -1)
    for j, (wn, ln, dt) in enumerate(zip(out[winner_col], out[loser_col], out[date_col])):
        found = -1
        for a in full_name_keys(wn):
            for b in full_name_keys(ln):
                for i in index.get((a, b), ()):
                    # archive dates are tournament start dates; odds dates are match dates
                    delta = (odates[i] - np.datetime64(dt)).astype("timedelta64[D]").astype(int)
                    if -2 <= delta <= 16:
                        found = i
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        hits[j] = found
    ok = hits >= 0
    for k in cols:
        vals = np.full(len(out), np.nan)
        vals[ok] = ovals[k][hits[ok]]
        out[k] = vals
    if series is not None:
        s = np.array([None] * len(out), dtype=object)
        s[ok] = series[hits[ok]]
        out["odds_series"] = s
    return out


def no_vig(odds_self, odds_other):
    a, b = 1.0 / np.asarray(odds_self, float), 1.0 / np.asarray(odds_other, float)
    return a / (a + b)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def blend_probability(p_model, p_market, blend: dict | None):
    """Benter-style combination of model and market (both for the same player)."""
    if not blend or p_market is None or not np.all(np.isfinite(np.asarray(p_market, float))):
        return p_model
    a = float(blend.get("model_coef", 0.0))
    b = float(blend.get("market_coef", 1.0))
    return sigmoid(a * logit(p_model) + b * logit(p_market))


def backtest(p_winner, odds_w, odds_l, *, min_ev=0.02, min_edge=0.0, kelly_fraction=0.25,
             kelly_cap=0.02, market_w=None) -> tuple[dict, pd.DataFrame]:
    """Simulate the app's betting rule on historical matches.

    ``p_winner`` is the probability the model gave to the player who actually won.
    Each match produces at most one bet (the side with higher EV).
    """
    p = np.asarray(p_winner, float)
    ow, ol = np.asarray(odds_w, float), np.asarray(odds_l, float)
    mw = no_vig(ow, ol) if market_w is None else np.asarray(market_w, float)
    ev_w, ev_l = p * ow - 1, (1 - p) * ol - 1
    back_w = ev_w >= ev_l
    ev = np.where(back_w, ev_w, ev_l)
    edge = np.where(back_w, p - mw, (1 - p) - (1 - mw))
    price = np.where(back_w, ow, ol)
    won = back_w  # the backed side won iff it was the actual winner
    sel = (ev >= min_ev) & (edge >= min_edge) & np.isfinite(ev)
    kelly = np.minimum(kelly_fraction * np.maximum(ev, 0) / np.maximum(price - 1, 1e-9), kelly_cap)
    bets = pd.DataFrame({
        "ev": ev[sel], "edge": edge[sel], "odds": price[sel], "won": won[sel].astype(int),
        "stake": kelly[sel],
    })
    if bets.empty:
        return {"bets": 0}, bets
    bets["flat_pl"] = np.where(bets.won == 1, bets.odds - 1, -1.0)
    bets["kelly_pl"] = bets.stake * bets.flat_pl
    n = len(bets)
    se = float(bets.flat_pl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    summary = {
        "bets": int(n),
        "share_of_matches": float(sel.mean()),
        "win_rate": float(bets.won.mean()),
        "avg_odds": float(bets.odds.mean()),
        "avg_claimed_ev": float(bets.ev.mean()),
        "flat_roi": float(bets.flat_pl.mean()),
        "flat_roi_se": se,
        "kelly_roi": float(bets.kelly_pl.sum() / bets.stake.sum()) if bets.stake.sum() > 0 else float("nan"),
    }
    return summary, bets
