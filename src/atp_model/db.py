
from datetime import datetime
from sqlalchemy import create_engine, String, Float, Integer, DateTime, UniqueConstraint
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
