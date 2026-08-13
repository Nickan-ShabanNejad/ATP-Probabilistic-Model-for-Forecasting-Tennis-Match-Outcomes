from __future__ import annotations

from typing import Any

import numpy as np


def decimal(value: Any) -> float | None:
    try:
        x = float(value)
        return x if np.isfinite(x) and x > 1.0 else None
    except Exception:
        return None


def no_vig_two_way(odds_a: Any, odds_b: Any) -> tuple[float, float] | None:
    """Return normalized two-way implied probabilities from decimal odds."""
    a, b = decimal(odds_a), decimal(odds_b)
    if a is None or b is None:
        return None
    ia, ib = 1.0 / a, 1.0 / b
    total = ia + ib
    if total <= 0:
        return None
    return ia / total, ib / total


def last_pre_match_quote(
    payload: dict,
    start_timestamp: int | float,
    *,
    bookmaker: str = "Pinnacle",
    market: str = "Full Time Result",
) -> dict | None:
    """Select the last quote strictly before the scheduled match start.

    This accepts Matchstat's ``last-10``-style odds-history response. Quotes at
    or after the event start are discarded so live/in-play prices never leak
    into a pre-match backtest.
    """
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    book = result.get(bookmaker, {}) if isinstance(result, dict) else {}
    quotes = book.get(market, []) if isinstance(book, dict) else []
    if not isinstance(quotes, list):
        return None

    valid = []
    start = float(start_timestamp)
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        try:
            ts = float(quote.get("sourceAddTime"))
        except Exception:
            continue
        o1, o2 = decimal(quote.get("od1")), decimal(quote.get("od2"))
        if ts < start and o1 is not None and o2 is not None:
            valid.append({**quote, "sourceAddTime": int(ts), "od1": o1, "od2": o2})
    return max(valid, key=lambda x: x["sourceAddTime"]) if valid else None


def safe_opening_quote(
    payload: dict,
    *,
    bookmaker: str = "Pinnacle",
    market: str = "Full Time Result",
) -> dict | None:
    """Extract the provider's explicit opening/start quote as a safe fallback."""
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    book = result.get(bookmaker, {}) if isinstance(result, dict) else {}
    market_data = book.get(market, {}) if isinstance(book, dict) else {}
    if not isinstance(market_data, dict):
        return None
    start = market_data.get("start")
    if not isinstance(start, dict):
        return None
    o1, o2 = decimal(start.get("od1")), decimal(start.get("od2"))
    if o1 is None or o2 is None:
        return None
    out = dict(start)
    out["od1"], out["od2"] = o1, o2
    return out
