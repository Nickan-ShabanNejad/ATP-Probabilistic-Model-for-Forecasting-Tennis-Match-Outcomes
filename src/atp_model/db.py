
from datetime import datetime
from sqlalchemy import (
    create_engine, String, Float, Integer, DateTime, inspect, text
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    match_key: Mapped[str] = mapped_column(String(300), index=True)
    player_a: Mapped[str] = mapped_column(String(150))
    player_b: Mapped[str] = mapped_column(String(150))
    surface: Mapped[str] = mapped_column(String(20))
    model_probability_a: Mapped[float] = mapped_column(Float)
    odds_a: Mapped[float] = mapped_column(Float)
    odds_b: Mapped[float] = mapped_column(Float)
    no_vig_probability_a: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    expected_value: Mapped[float] = mapped_column(Float)
    closing_odds_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_no_vig_probability_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stake: Mapped[float] = mapped_column(Float, default=0.0)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)


class DataRun(Base):
    __tablename__ = "data_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    latest_match_date: Mapped[str] = mapped_column(String(20))
    matches_loaded: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(String(500), default="")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)

    # create_all does not add columns to an existing table, so migrate safely.
    existing = {column["name"] for column in inspect(engine).get_columns("predictions")}
    additions = {
        "closing_odds_b": "FLOAT",
        "closing_no_vig_probability_a": "FLOAT",
        "probability_clv": "FLOAT",
        "price_clv": "FLOAT",
    }
    with engine.begin() as connection:
        for column, sql_type in additions.items():
            if column not in existing:
                connection.execute(
                    text(f"ALTER TABLE predictions ADD COLUMN {column} {sql_type}")
                )
