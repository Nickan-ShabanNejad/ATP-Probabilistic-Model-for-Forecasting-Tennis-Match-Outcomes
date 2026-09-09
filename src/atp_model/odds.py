from __future__ import annotations

from typing import Any, Iterator

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
    """Select the last quote strictly before the scheduled match start."""
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


def _iter_rows(obj: Any, bookmaker: str = "", market: str = "") -> Iterator[tuple[dict, str, str]]:
    """Walk changing Matchstat odds response shapes while retaining parent context."""
    if isinstance(obj, dict):
        row_book = str(obj.get("bookmaker") or obj.get("bookmakerName") or obj.get("book") or bookmaker or "")
        row_market = str(obj.get("market") or obj.get("marketName") or obj.get("market_name") or market or "")
        yield obj, row_book, row_market
        for key, value in obj.items():
            next_book, next_market = row_book, row_market
            key_text = str(key)
            if "pinnacle" in key_text.casefold():
                next_book = "Pinnacle"
            low = key_text.casefold()
            if any(token in low for token in ("full time result", "match winner", "moneyline", "total sets", "sets total", "total set")):
                next_market = key_text
            yield from _iter_rows(value, next_book, next_market)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_rows(value, bookmaker, market)


def _two_way_from_row(row: dict) -> tuple[float, float] | None:
    a = decimal(row.get("od1") or row.get("odds1") or row.get("outcome1Odds") or row.get("homeOdds"))
    b = decimal(row.get("od2") or row.get("odds2") or row.get("outcome2Odds") or row.get("awayOdds"))
    return (a, b) if a is not None and b is not None else None


def pinnacle_moneyline_from_payload(payload: Any) -> tuple[float, float] | None:
    """Return latest Pinnacle two-way match-winner odds from Matchstat response variants."""
    if not isinstance(payload, (dict, list)):
        return None
    for row, book, market in _iter_rows(payload):
        if "pinnacle" not in book.casefold():
            continue
        market_text = market.casefold()
        # A blank market is accepted only when the row explicitly contains a two-way
        # Pinnacle quote (common on compare endpoint result rows).
        if market_text and not any(x in market_text for x in ("full time result", "match winner", "moneyline", "winner")):
            continue
        pair = _two_way_from_row(row)
        if pair:
            return pair
    return None


def pinnacle_total_sets_35_from_payload(payload: Any) -> tuple[float, float] | None:
    """Return (Over 3.5, Under 3.5) Pinnacle decimal odds when Matchstat exposes them."""
    if not isinstance(payload, (dict, list)):
        return None

    flat_pairs: list[tuple[float, float]] = []
    singles: dict[str, float] = {}
    for row, book, parent_market in _iter_rows(payload):
        if "pinnacle" not in book.casefold():
            continue
        text = " ".join(
            str(x or "")
            for x in (
                parent_market,
                row.get("market"), row.get("marketName"), row.get("market_name"),
                row.get("name"), row.get("label"), row.get("selection"),
                row.get("outcome"), row.get("outcomeName"),
            )
        ).casefold()
        line = row.get("line", row.get("handicap", row.get("total", row.get("value"))))
        try:
            line_num = float(line) if line not in (None, "") else None
        except Exception:
            line_num = None
        is_sets = "set" in text
        is_35 = line_num == 3.5 or "3.5" in text
        if not (is_sets and is_35):
            continue

        pair = _two_way_from_row(row)
        if pair:
            o1, o2 = pair
            outcome1 = str(row.get("outcome1") or row.get("selection1") or row.get("label1") or "").casefold()
            outcome2 = str(row.get("outcome2") or row.get("selection2") or row.get("label2") or "").casefold()
            if "under" in outcome1 or "over" in outcome2:
                flat_pairs.append((o2, o1))
            else:
                # Most two-way total markets expose outcome1=Over, outcome2=Under.
                flat_pairs.append((o1, o2))
            continue

        odds = decimal(row.get("odds") or row.get("price") or row.get("decimalOdds"))
        if odds:
            if "over" in text:
                singles["over"] = odds
            elif "under" in text:
                singles["under"] = odds

    if "over" in singles and "under" in singles:
        return float(singles["over"]), float(singles["under"])
    return flat_pairs[0] if flat_pairs else None
