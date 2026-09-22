@staticmethod
def _moneyline(row: dict) -> tuple[float, float] | None:
    periods = row.get("periods") or {}
    game = periods.get("num_0") if isinstance(periods, dict) else None

    if not isinstance(game, dict):
        return None

    ml = game.get("money_line") or game.get("moneyline") or {}

    if not isinstance(ml, dict):
        return None

    try:
        home = float(ml.get("home"))
        away = float(ml.get("away"))
    except (TypeError, ValueError):
        return None

    if home <= 1.0 or away <= 1.0:
        return None

    return home, away


@classmethod
def _moneyline_from_payload(cls, payload):
    if isinstance(payload, dict):
        pair = cls._moneyline(payload)

        if pair is not None:
            return pair

        events = payload.get("events")

        if isinstance(events, list):
            for row in events:
                if isinstance(row, dict):
                    pair = cls._moneyline(row)

                    if pair is not None:
                        return pair

        for key in ("event", "result", "data"):
            child = payload.get(key)

            if isinstance(child, dict):
                pair = cls._moneyline_from_payload(child)

                if pair is not None:
                    return pair

    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                pair = cls._moneyline_from_payload(row)

                if pair is not None:
                    return pair

    return None
