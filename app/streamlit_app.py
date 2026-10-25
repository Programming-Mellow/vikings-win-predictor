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
from vikpred.llm.learner import LEVELS, LearnerStore  # noqa: E402
from vikpred.llm.provider import credentials_available, load_config, make_client  # noqa: E402
from vikpred.llm.toolbox import MCPToolbox  # noqa: E402
from vikpred.tracker import summary, tracker_frame  # noqa: E402

API_URL = os.environ.get("VIKPRED_API_URL", "http://localhost:8000")
PURPLE, GOLD, GREY = "#4F2683", "#FFC62F", "#9E9E9E"
PURPLE_LIGHT = "#9B6FE0"  # brand purple lifted for contrast on dark backgrounds
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


def tracker_section(team: str):
    st.subheader("Season tracker: predictions vs. results")
    live_only = st.toggle("Live predictions only", help="Exclude predictions made after kickoff (backfilled).")
    frame = tracker_frame(api_get(f"/teams/{team}/games"), live_only=live_only)
    if frame.empty:
        st.info("No scored games yet with live predictions. Live calls start counting from the first game "
                "the weekly job predicts before kickoff." if live_only else "No scored games yet.")
        return
    s = summary(frame)
    c1, c2 = st.columns(2)
    c1.metric("Model: picks · Brier", f"{s['model_correct']}/{s['games']} · {s['model_brier']:.3f}")
    c2.metric("Vegas: picks · Brier", f"{s['vegas_correct']}/{s['games']} · {s['vegas_brier']:.3f}",
              help="Brier score = average squared gap between the predicted chance and the result. Lower is better.")

    frame["game"] = frame.apply(lambda g: f"Wk {g.week} {'vs' if g.venue == 'home' else '@'} {g.opponent}", axis=1)
    order = frame.game.tolist()
    probs = frame.melt(id_vars=["game", "result"], value_vars=["model_prob", "vegas_prob"],
                       var_name="source", value_name="prob")
    probs["source"] = probs.source.map({"model_prob": "Model", "vegas_prob": "Vegas"})
    # Model and Vegas often agree to within a point or two, so the dots sit side by
    # side (xOffset) rather than on top of each other. The lighter purple reads on dark themes.
    colors = alt.Scale(domain=["Model", "Vegas"], range=[PURPLE_LIGHT, GOLD])
    points = alt.Chart(probs).mark_point(filled=True, size=110, opacity=1).encode(
        x=alt.X("game:N", sort=order, title=None),
        xOffset=alt.XOffset("source:N", scale=alt.Scale(paddingOuter=0.6)),
        y=alt.Y("prob:Q", title=f"Predicted {team} win chance", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        color=alt.Color("source:N", scale=colors, title=None),
        tooltip=["game", "source", alt.Tooltip("prob:Q", format=".0%"), "result"])
    labels = alt.Chart(frame).mark_text(dy=-6, fontWeight="bold", fontSize=13).encode(
        x=alt.X("game:N", sort=order), y=alt.value(0), text="result:N",
        color=alt.condition(alt.datum.result == "W", alt.value(PURPLE_LIGHT), alt.value(GREY)))
    coin = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q")
    st.altair_chart((coin + points + labels).properties(height=260), width="stretch")
    st.caption("Dots are each pre-game call; letters on top are the actual result. The dashed line is a coin flip.")

    brier = frame.melt(id_vars=["game"], value_vars=["cum_model_brier", "cum_vegas_brier"],
                       var_name="source", value_name="brier")
    brier["source"] = brier.source.map({"cum_model_brier": "Model", "cum_vegas_brier": "Vegas"})
    st.altair_chart(alt.Chart(brier).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("game:N", sort=order, title=None),
        y=alt.Y("brier:Q", title="Running Brier"),
        color=alt.Color("source:N", scale=colors, title=None),
        strokeDash=alt.StrokeDash("source:N", scale=alt.Scale(domain=["Model", "Vegas"], range=[[1, 0], [6, 4]]),
                                  legend=None),
        tooltip=["game", "source", alt.Tooltip("brier:Q", format=".3f")]).properties(height=220), width="stretch")
    st.caption("Running Brier score: the average squared gap between each call and the result so far. Lower is better.")


# -- chat ------------------------------------------------------------------------------------

def get_learner() -> LearnerStore:
    if "learner" not in st.session_state:
        st.session_state.learner = LearnerStore()
    return st.session_state.learner


def get_agent() -> ChatAgent | None:
    config = load_config()
    if not credentials_available(config):
        return None
    if "agent" not in st.session_state:
        toolbox = MCPToolbox(env={**os.environ, "VIKPRED_API_URL": API_URL})
        st.session_state.agent = ChatAgent(make_client(config), config, toolbox.tools, toolbox.call,
                                           learner=get_learner())
        st.session_state.exchanges = []
    return st.session_state.agent


def render_exchange(ex: dict):
    with st.chat_message("user"):
        st.markdown(ex["user"])
    with st.chat_message("assistant"):
        for line in ex.get("tools", []):
            st.caption(line)
        st.markdown(ex["assistant"])


def stream_answer(agent: ChatAgent, prompt: str) -> dict:
    """Render a new exchange while it streams; returns it for the log."""
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        tool_lines, text = [], ""
        status_area, text_area = st.container(), st.empty()
        note, note_slot = "", None
        for event in agent.respond(prompt):
            if event.kind == "text":
                text += event.text
                text_area.markdown(text + "▌")
            elif event.kind == "progress":
                # A progress note streams in pieces; keep it on one status line.
                if note_slot is None:
                    note_slot = status_area.empty()
                note += event.text
                note_slot.caption(f"💭 {note}")
            elif event.kind == "tool_start":
                if note:
                    tool_lines.append(f"💭 {note}")
                note, note_slot = "", None
                line = describe_tool_call(event.tool, event.data)
                tool_lines.append(f"🔧 {line}")
                status_area.caption(f"🔧 {line}…")
            elif event.kind == "tool_end" and event.data and event.data.get("is_error"):
                status_area.caption(f"⚠️ {event.tool} returned an error; the assistant will adjust.")
            elif event.kind in ("notice", "error"):
                text += f"\n\n_{event.text}_"
        text_area.markdown(text or "_No answer._")
    return {"user": prompt, "assistant": text, "tools": tool_lines}


def profile_panel(learner: LearnerStore):
    profile = learner.profile
    with st.expander("What the assistant knows about you"):
        st.markdown(f"**Level:** {LEVELS[profile.level][0]}")
        st.markdown(f"**Knows:** {', '.join(profile.known) or '_nothing recorded yet_'}")
        st.markdown(f"**Learning:** {', '.join(profile.learning) or '_nothing yet_'}")
        st.markdown(f"**Interests:** {', '.join(profile.interests) or '_nothing recorded yet_'}")
        st.caption("Saved on this computer in data/learner_profile.json and updated as you chat.")
        if st.button("Reset what it knows", type="secondary"):
            learner.reset()
            st.rerun()


def chat_sidebar():
    with st.sidebar:
        st.header("Ask the assistant")
        agent = get_agent()
        if agent is None:
            st.info("Add your Anthropic API key to a `.env` file in the project folder to turn on the chat:\n\n"
                    "`ANTHROPIC_API_KEY=sk-ant-...`\n\nThen restart Streamlit. The dashboard works without it.")
            return
        learner = get_learner()
        st.caption("Powered by Claude, using this project's model through MCP tools. Not betting advice.")

        level = st.segmented_control(
            "Your football level", options=list(LEVELS), format_func=lambda k: LEVELS[k][0],
            default=learner.profile.level,
            help="Answers adjust to this, and to what the assistant learns you already know.")
        if level and level != learner.profile.level:
            learner.set_level(level)

        # The input comes first so it stays pinned at the top; the conversation
        # scrolls in its own box below, newest exchange first.
        prompt = st.chat_input("e.g. Why is the model confident about this week?")
        conversation = st.container(height=560, border=False)
        with conversation:
            if prompt:
                st.session_state.exchanges.append(stream_answer(agent, prompt))
                earlier = st.session_state.exchanges[:-1]
            else:
                earlier = st.session_state.exchanges
            if not st.session_state.exchanges:
                st.caption("Ask about this week's game, the season, how the model works, or anything NFL.")
            for ex in reversed(earlier):
                render_exchange(ex)
        profile_panel(learner)


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
    st.divider()
    tracker_section(team)
    chat_sidebar()


main()
