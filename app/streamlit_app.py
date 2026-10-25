"""Vikings Win Predictor: dashboard + Claude chat assistant.

Run (with the API running on :8000):
    streamlit run app/streamlit_app.py
"""

import os
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from vikpred.llm.agent import ChatAgent, describe_tool_call  # noqa: E402
from vikpred.llm.provider import credentials_available, load_config, make_client  # noqa: E402
from vikpred.llm.toolbox import MCPToolbox  # noqa: E402

API_URL = os.environ.get("VIKPRED_API_URL", "http://localhost:8000")
PURPLE, GOLD, GREY = "#4F2683", "#FFC62F", "#9E9E9E"
TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU",
         "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT",
         "SEA", "SF", "TB", "TEN", "WAS"]
MODEL_NAMES = {"xgb_full": "Stats + Vegas model", "xgb_stats": "Stats-only model"}

st.set_page_config(page_title="Vikings Win Predictor", page_icon="🏈", layout="wide")


# -- data -------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, **params):
    resp = httpx.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def pct(p) -> str:
    return "—" if p is None else f"{p:.0%}"


# -- dashboard pieces ---------------------------------------------------------------------

def factor_chart(factors: list[dict]):
    df = pd.DataFrame(factors)
    df["direction"] = df.impact.map(lambda x: "Helps" if x >= 0 else "Hurts")
    return (alt.Chart(df).mark_bar().encode(
        x=alt.X("impact:Q", title="Push on win chance (log-odds)"),
        y=alt.Y("label:N", sort=alt.EncodingSortField("impact", op="sum", order="descending"), title=None,
                axis=alt.Axis(labelLimit=380)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Helps", "Hurts"], range=[PURPLE, GOLD]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Factor"), alt.Tooltip("value:Q", title="Value", format=".3f"),
                 alt.Tooltip("impact:Q", title="Impact", format="+.3f")],
    ).properties(height=36 * len(df)))


def next_game_section(team: str):
    try:
        game = api_get(f"/teams/{team}/next", top_k=6)
    except httpx.HTTPStatusError as exc:
        st.info(exc.response.json().get("detail", "No upcoming game."))
        return
    where = "vs" if game["venue"] == "home" else "@"
    st.subheader(f"Week {game['week']}: {team} {where} {game['opponent']}")
    st.caption(f"{game['gameday']} · predictions updated {game['predicted_at'][:16].replace('T', ' ')} UTC")

    c1, c2, c3 = st.columns(3)
    c1.metric("Stats + Vegas model", pct(game["win_prob"]),
              help="XGBoost on team stats plus the Vegas line. Statistically ties Vegas on 2024-25 test games.")
    c2.metric("Stats-only model", pct(game["win_prob_stats"]),
              help="XGBoost on team stats only: an independent, football-only view.")
    c3.metric("Vegas", pct(game["vegas_prob"]), help="Moneyline win probability with the bookmaker margin removed.")

    tabs = st.tabs([MODEL_NAMES[p["model"]] for p in game["predictions"]])
    for tab, pred in zip(tabs, game["predictions"]):
        with tab:
            if not pred["available"]:
                st.info(pred["reason"])
                continue
            st.altair_chart(factor_chart(pred["top_factors"]), width="stretch")
            st.caption("Purple bars raise the win chance, gold bars lower it. Values compare the two teams.")


def season_section(team: str):
    games = pd.DataFrame(api_get(f"/teams/{team}/games"))
    record = api_get(f"/teams/{team}/record")
    st.subheader(f"{record['season']} season")
    c1, c2, c3 = st.columns(3)
    c1.metric("Record", f"{record['wins']}-{record['losses']}" + (f"-{record['ties']}" if record["ties"] else ""))
    if record["games_scored"]:
        n = record["games_scored"]
        c2.metric("Model picked winner", f"{record['model_correct']}/{n}", help=f"Brier {record['model_brier']}")
        c3.metric("Vegas picked winner", f"{record['vegas_correct']}/{n}", help=f"Brier {record['vegas_brier']}")

    played = games.result.notna()
    table = pd.DataFrame({
        "Week": games.week,
        "Opponent": games.apply(lambda g: ("vs " if g.venue == "home" else "@ ") + g.opponent, axis=1),
        "Stats + Vegas": games.win_prob.map(lambda p: pct(None if pd.isna(p) else p)),
        "Stats only": games.win_prob_stats.map(lambda p: pct(None if pd.isna(p) else p)),
        "Vegas": games.vegas_prob.map(lambda p: pct(None if pd.isna(p) else p)),
        "Result": (games.result + " " + games.score).where(played, ""),
        "Live call": games.backfilled.map({True: "no (backfilled)", False: "yes"}).where(played, ""),
    })
    st.dataframe(table, hide_index=True, width="stretch")


# -- chat ------------------------------------------------------------------------------------

def get_agent() -> ChatAgent | None:
    config = load_config()
    if not credentials_available(config):
        return None
    if "agent" not in st.session_state:
        toolbox = MCPToolbox(env={**os.environ, "VIKPRED_API_URL": API_URL})
        st.session_state.agent = ChatAgent(make_client(config), config, toolbox.tools, toolbox.call)
        st.session_state.chat_log = []
    return st.session_state.agent


def chat_sidebar():
    with st.sidebar:
        st.header("Ask the assistant")
        agent = get_agent()
        if agent is None:
            st.info("Add your Anthropic API key to a `.env` file in the project folder to turn on the chat:\n\n"
                    "`ANTHROPIC_API_KEY=sk-ant-...`\n\nThen restart Streamlit. The dashboard works without it.")
            return
        st.caption("Powered by Claude, using this project's model through MCP tools. Not betting advice.")

        for msg in st.session_state.chat_log:
            with st.chat_message(msg["role"]):
                for line in msg.get("tools", []):
                    st.caption(f"🔧 {line}")
                st.markdown(msg["content"])

        prompt = st.chat_input("e.g. Why is the model confident about this week?")
        if not prompt:
            return
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            tool_lines, text = [], ""
            status_area, text_area = st.container(), st.empty()
            for event in agent.respond(prompt):
                if event.kind == "text":
                    text += event.text
                    text_area.markdown(text + "▌")
                elif event.kind == "tool_start":
                    line = describe_tool_call(event.tool, event.data)
                    tool_lines.append(line)
                    status_area.caption(f"🔧 {line}…")
                elif event.kind == "tool_end" and event.data and event.data.get("is_error"):
                    status_area.caption(f"⚠️ {event.tool} returned an error; the assistant will adjust.")
                elif event.kind in ("notice", "error"):
                    text += f"\n\n_{event.text}_"
            text_area.markdown(text or "_No answer._")
        st.session_state.chat_log.append({"role": "assistant", "content": text, "tools": tool_lines})


# -- page ------------------------------------------------------------------------------------

def main():
    st.title("🏈 Vikings Win Predictor")
    st.caption("Pre-game win probabilities from XGBoost models trained on every NFL game since 2010, "
               "explained with SHAP. For fun and learning, not betting advice.")
    try:
        api_get("/health")
    except httpx.HTTPError:
        st.error(f"Can't reach the prediction API at {API_URL}. Start it with "
                 "`uvicorn vikpred.api.main:app`, then refresh.")
        st.stop()

    team = st.selectbox("Team", TEAMS, index=TEAMS.index("MIN"))
    next_game_section(team)
    st.divider()
    season_section(team)
    chat_sidebar()


main()
