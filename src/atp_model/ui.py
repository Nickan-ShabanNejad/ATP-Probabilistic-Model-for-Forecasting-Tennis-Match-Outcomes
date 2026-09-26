"""Shared look-and-feel for every Streamlit page (v0.4)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from .config import ROOT

APP_VERSION = "v0.4.0"

LEVEL_LABELS = {1.5: "Davis Cup", 2.0: "ATP 250", 3.0: "ATP 500", 4.0: "Masters 1000", 4.5: "ATP Finals", 5.0: "Grand Slam"}
SURFACE_EMOJI = {"Hard": "🟦", "Clay": "🟧", "Grass": "🟩"}

CSS = """
<style>
:root {
  --ok: #3ddc84; --warn: #fab219; --bad: #e66767; --muted: #8b949e;
  --card: #161b22; --card-2: #1c2230; --line: #2a313c; --blue: #3987e5; --orange: #d95926;
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.app-hero { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.6rem; }
.app-hero h1 { font-size: 2.0rem; margin: 0; }
.app-hero .sub { color: var(--muted); font-size: .95rem; margin-top:.2rem; }
.badge { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;
         border:1px solid var(--line); background: var(--card-2); color:#c9d1d9; margin-right:.3rem; white-space:nowrap; }
.badge.ok { background: rgba(61,220,132,.12); color: var(--ok); border-color: rgba(61,220,132,.35); }
.badge.warn { background: rgba(250,178,25,.12); color: var(--warn); border-color: rgba(250,178,25,.35); }
.badge.bad { background: rgba(230,103,103,.12); color: var(--bad); border-color: rgba(230,103,103,.35); }
.badge.blue { background: rgba(57,135,229,.12); color: #79b0f2; border-color: rgba(57,135,229,.35); }
.kpi { background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; height:100%; }
.kpi .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { font-size: 1.55rem; font-weight: 700; margin-top:.15rem; }
.kpi .hint { color: var(--muted); font-size: .8rem; margin-top:.1rem; }
.bet-card { background: linear-gradient(180deg, rgba(61,220,132,.08), rgba(61,220,132,.02)); border:1px solid rgba(61,220,132,.35);
            border-radius: 14px; padding: .9rem 1.1rem; margin-bottom:.2rem; }
.bet-card .pick { font-size: 1.2rem; font-weight: 700; }
.bet-card .meta { color: var(--muted); font-size: .85rem; margin-top: .15rem; }
.bet-card .nums { display:flex; gap:1.4rem; flex-wrap:wrap; margin-top:.55rem; }
.bet-card .nums div span { display:block; color: var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }
.bet-card .nums div b { font-size: 1.05rem; }
.match-row .players { font-weight: 600; line-height:1.5; }
.probbar { height: 8px; border-radius: 999px; background: #2a313c; overflow:hidden; display:flex; margin:.25rem 0 .1rem; }
.probbar .a { background: var(--blue); }
.probbar .b { background: var(--orange); }
.probbar .gap { width:2px; background: var(--card); }
.small-muted { color: var(--muted); font-size: .82rem; }
.empty { border:1px dashed var(--line); border-radius: 14px; padding: 1.2rem 1.3rem; color:#c9d1d9; background: var(--card); }
div[data-testid="stExpander"] details { border-radius: 12px; }
</style>
"""


def setup_page(title: str, icon: str = "🎾") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", badges: list[tuple[str, str]] | None = None) -> None:
    b = "".join(f'<span class="badge {html.escape(k)}">{html.escape(t)}</span>' for t, k in (badges or []))
    st.markdown(
        f'<div class="app-hero"><div><h1>{html.escape(title)}</h1>'
        f'<div class="sub">{html.escape(subtitle)}</div></div><div>{b}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, hint: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="hint">{html.escape(hint)}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{html.escape(str(text))}</span>'


def prob_bar(p_a: float) -> str:
    try:
        pa = max(0.0, min(1.0, float(p_a)))
    except Exception:
        return ""
    return (f'<div class="probbar"><div class="a" style="width:{pa*100:.1f}%"></div>'
            f'<div class="gap"></div><div class="b" style="flex:1"></div></div>')


def level_label(level) -> str:
    try:
        return LEVEL_LABELS.get(float(level), f"Level {level}")
    except Exception:
        return str(level)


@st.cache_data(ttl=600)
def load_metrics() -> dict:
    path = ROOT / "data/generated/metrics.json"
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def esc(value) -> str:
    return html.escape(str(value))
