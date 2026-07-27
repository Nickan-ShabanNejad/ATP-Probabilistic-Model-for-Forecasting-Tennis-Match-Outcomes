
from .db import SessionLocal, Prediction, init_db


def save_prediction(result, odds_a, odds_b, stake=0.0):
    init_db()
    key = f"{result['player_a']}|{result['player_b']}|{result['surface']}"
    with SessionLocal() as db:
        row = Prediction(
            match_key=key,
            player_a=result["player_a"],
            player_b=result["player_b"],
            surface=result["surface"],
            model_probability_a=result["probability_a"],
            odds_a=odds_a,
            odds_b=odds_b,
            no_vig_probability_a=result["market_probability_a"],
            edge=result["edge"],
            expected_value=result["ev"],
            stake=stake,
        )
        db.add(row)
        db.commit()
        return row.id


def get_predictions():
    init_db()
    with SessionLocal() as db:
        return db.query(Prediction).order_by(Prediction.created_at.desc()).all()


def settle_prediction(
    prediction_id,
    result_a,
    closing_odds_a=None,
    closing_odds_b=None,
):
    init_db()
    with SessionLocal() as db:
        row = db.get(Prediction, prediction_id)
        if not row:
            raise ValueError("Prediction not found")

        row.result_a = int(result_a)
        row.closing_odds_a = closing_odds_a
        row.closing_odds_b = closing_odds_b
        row.profit = (row.stake * (row.odds_a - 1)) if result_a else -row.stake

        if (
            closing_odds_a is not None
            and closing_odds_b is not None
            and closing_odds_a > 1
            and closing_odds_b > 1
        ):
            raw_a = 1 / closing_odds_a
            raw_b = 1 / closing_odds_b
            close_no_vig_a = raw_a / (raw_a + raw_b)
            row.closing_no_vig_probability_a = close_no_vig_a

            # Positive probability CLV means the closing market assigned a larger
            # win probability to the saved side than the opening market did.
            row.probability_clv = (
                close_no_vig_a - row.no_vig_probability_a
            )

            # Positive price CLV means the taken decimal price beat the close.
            row.price_clv = (row.odds_a / closing_odds_a) - 1

        db.commit()
