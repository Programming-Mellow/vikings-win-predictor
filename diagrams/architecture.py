"""
Architecture: Vikings Win Predictor - Full System
Last Updated: 2026-10-25
Description: Data flows left to right: nflverse data feeds the offline ML pipeline
             (features -> XGBoost + SHAP, tracked in MLflow); a scheduled job writes
             locked pre-game predictions; the FastAPI service serves them to the
             Streamlit dashboard and, through the MCP server, to the Claude chat agent.
             Everything right of the pipeline runs under Docker Compose.
Layers:      Ingestion = nflverse via nflreadpy; Storage = Parquet files + model
             artifacts on a shared volume; Compute = Python pipeline, scheduled job,
             XGBoost; API/access = FastAPI, MCP server, Streamlit; Monitoring =
             MLflow experiment tracking and API /health checks (no runtime metrics
             stack - out of scope for a single-host deployment; CI is shown as a
             README badge rather than in the runtime diagram).
Regenerate:  python diagrams/architecture.py   (needs Graphviz + `pip install diagrams`)
Output:      docs/architecture.png
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Client, User
from diagrams.onprem.compute import Server
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import FastAPI
from diagrams.programming.language import Python

ROOT = Path(__file__).resolve().parents[1]
PURPLE, GOLD, GREY = "#4F2683", "#C99700", "#8A8A8A"

graph_attr = {"rankdir": "LR", "fontsize": "26", "fontname": "Helvetica", "labelloc": "t",
              "pad": "0.5", "nodesep": "0.55", "ranksep": "0.9", "splines": "spline", "bgcolor": "white"}
node_attr = {"fontsize": "13", "fontname": "Helvetica"}
edge_attr = {"fontsize": "11", "fontname": "Helvetica", "color": GREY, "penwidth": "1.6"}


def cluster(label: str, color: str) -> dict:
    return {"bgcolor": color, "fontsize": "15", "fontname": "Helvetica-Bold", "margin": "18",
            "style": "rounded", "pencolor": "#C8C8C8"}


with Diagram("Vikings Win Predictor: architecture", filename=str(ROOT / "docs" / "architecture"),
             outformat="png", graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr, show=False):
    nflverse = Github("nflverse\nschedules · play-by-play\ninjuries · betting lines")

    with Cluster("1 · Offline ML pipeline", graph_attr=cluster("", "#EEF3FB")):
        ingest = Python("ingest\n(polars)")
        features = Python("pre-game features\n(leakage-safe)")
        train = Python("XGBoost + SHAP\nrolling-season CV")
        with Cluster("experiment tracking", graph_attr={"bgcolor": "#2F3B48", "fontcolor": "white",
                                                        "style": "rounded", "fontsize": "11", "margin": "10"}):
            mlflow = Mlflow("")
        models = Storage("model\nartifacts")

    with Cluster("2 · Serving (Docker Compose)", graph_attr=cluster("", "#F6F1FC")):
        job = Python("scheduled job\nlocks pre-game calls")
        preds = Storage("predictions\n+ SHAP values")
        api = FastAPI("FastAPI\nprediction API")
        mcp = Python("MCP server\n6 tools")
        with Cluster("Streamlit app", graph_attr={"bgcolor": "#FFF8E1", "style": "rounded",
                                                  "fontsize": "13", "pencolor": "#E6D9A8"}):
            dashboard = Client("dashboard +\nseason tracker")
            agent = Python("Claude chat agent\n+ learner profile")

    claude = Server("Claude Sonnet 5.5\nAnthropic API")
    fan = User("you")
    mcp_clients = Client("Claude Desktop /\nClaude Code")

    nflverse >> ingest >> features >> train >> models
    train >> Edge(style="dashed", label="runs") >> mlflow
    nflverse >> Edge(label="refresh") >> job
    models >> job >> preds >> api

    api >> Edge(label="HTTP", color=PURPLE) >> dashboard
    api >> Edge(label="HTTP", color=GOLD) >> mcp
    mcp >> Edge(label="tools (stdio)", color=GOLD) >> agent
    agent >> Edge(label="streamed chat", color=GOLD, dir="both") >> claude
    dashboard >> Edge(color=PURPLE) >> fan
    agent >> Edge(color=GOLD) >> fan
    mcp >> Edge(style="dashed", label="any MCP client") >> mcp_clients
