"""
Hindsight — Dashboard analytics.

Public API:
    compute_stats()            -> dict (raw aggregations for plotting + LLM)
    suggest(stats)             -> list[str] (LLM-generated actionable advice)
    emotion_timeline_fig(df)   -> plotly.Figure (line chart over time)
    topic_bars_fig(stats)      -> plotly.Figure (mood-boosters vs mood-killers)

All metrics are derived ONLY from the user's own entries — no external data.
"""
from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from pipeline import load_entries, GROQ_MODEL

load_dotenv()
PROMPTS = Path(__file__).parent / "prompts"

POSITIVE_EMOTIONS = ("joy", "neutral")
NEGATIVE_EMOTIONS = ("sadness", "anger", "fear", "disgust")


# ============================================================ helpers
def _entries_df() -> pd.DataFrame:
    """Flatten entries into a tidy DataFrame for analysis."""
    entries = load_entries()
    if not entries:
        return pd.DataFrame()

    rows = []
    for e in entries:
        row = {
            "id":          e["id"],
            "timestamp":   pd.to_datetime(e["timestamp"]),
            "text":        e["text"],
            "source":      e["source"],
            "topics":      e.get("topics", []),
            "quick_mood":  e.get("quick_mood"),
        }
        # explode emotions into columns
        for k, v in (e.get("emotions") or {}).items():
            row[f"emo_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"]    = df["timestamp"].dt.date
    df["dow"]     = df["timestamp"].dt.day_name()
    df["dow_num"] = df["timestamp"].dt.dayofweek

    # composite "valence" — positive minus negative, in [-1, 1]
    pos_cols = [c for c in df.columns if c.replace("emo_", "") in POSITIVE_EMOTIONS]
    neg_cols = [c for c in df.columns if c.replace("emo_", "") in NEGATIVE_EMOTIONS]
    df["pos"] = df[pos_cols].sum(axis=1) if pos_cols else 0.0
    df["neg"] = df[neg_cols].sum(axis=1) if neg_cols else 0.0
    df["valence"] = df["pos"] - df["neg"]      # ~[-1, +1]

    return df.sort_values("timestamp").reset_index(drop=True)


# ============================================================ stats
def compute_stats() -> dict:
    """Return a structured summary used by both the UI and the LLM."""
    df = _entries_df()
    if df.empty:
        return {"n_entries": 0}

    # Mood-by-topic
    topic_rows = []
    for _, row in df.iterrows():
        for t in (row["topics"] or []):
            topic_rows.append({"topic": t, "valence": row["valence"]})
    tdf = pd.DataFrame(topic_rows)

    if tdf.empty:
        topics_pos, topics_neg, top_topics = [], [], []
    else:
        agg = (tdf.groupby("topic")
                  .agg(count=("valence", "size"), mean_valence=("valence", "mean"))
                  .reset_index())
        # only consider topics that appeared >=2 times to avoid noise
        agg = agg[agg["count"] >= 2].sort_values("mean_valence", ascending=False)
        topics_pos = agg.head(5).to_dict("records")
        topics_neg = agg.tail(5).iloc[::-1].to_dict("records")
        top_topics = (tdf["topic"].value_counts()
                                  .head(10)
                                  .reset_index()
                                  .rename(columns={"count": "n", "topic": "topic"})
                                  .to_dict("records"))

    # Day-of-week pattern
    dow_agg = (df.groupby(["dow_num", "dow"])
                 .agg(mean_valence=("valence", "mean"), n=("valence", "size"))
                 .reset_index()
                 .sort_values("dow_num"))

    # Last-7 vs previous-7 trend
    last7 = df.tail(7)["valence"].mean() if len(df) >= 7 else df["valence"].mean()
    prev7 = df.iloc[-14:-7]["valence"].mean() if len(df) >= 14 else None

    return {
        "n_entries":   int(len(df)),
        "date_range":  (str(df["date"].min()), str(df["date"].max())),
        "avg_valence": round(float(df["valence"].mean()), 3),
        "trend": {
            "last_7":   round(float(last7), 3),
            "prev_7":   round(float(prev7), 3) if prev7 is not None else None,
        },
        "top_topics":  top_topics,
        "topics_pos":  [{"topic": r["topic"], "n": int(r["count"]),
                          "mean_valence": round(float(r["mean_valence"]), 3)}
                         for r in topics_pos],
        "topics_neg":  [{"topic": r["topic"], "n": int(r["count"]),
                          "mean_valence": round(float(r["mean_valence"]), 3)}
                         for r in topics_neg],
        "day_of_week": dow_agg.to_dict("records"),
    }


# ============================================================ LLM suggestions
@lru_cache(maxsize=1)
def _suggest_prompt() -> str:
    return (PROMPTS / "suggestions.txt").read_text()


def suggest(stats: dict) -> list[str]:
    """Turn the stats summary into 3-5 actionable suggestions via Groq."""
    if stats.get("n_entries", 0) < 5:
        return ["Not enough entries yet. Record a few more days, then come back."]

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Compact summary for the prompt
    summary_lines = [
        f"Total entries: {stats['n_entries']}",
        f"Date range: {stats['date_range'][0]} → {stats['date_range'][1]}",
        f"Average valence (positive−negative emotion): {stats['avg_valence']}",
        f"Recent 7-day valence: {stats['trend']['last_7']}",
    ]
    if stats["trend"]["prev_7"] is not None:
        summary_lines.append(f"Previous 7-day valence: {stats['trend']['prev_7']}")

    summary_lines.append("\nTop mood-positive topics (n>=2):")
    for r in stats["topics_pos"]:
        summary_lines.append(f"  • {r['topic']:<14}  n={r['n']:>2}  mean_valence={r['mean_valence']:+.3f}")

    summary_lines.append("\nTop mood-negative topics (n>=2):")
    for r in stats["topics_neg"]:
        summary_lines.append(f"  • {r['topic']:<14}  n={r['n']:>2}  mean_valence={r['mean_valence']:+.3f}")

    summary_lines.append("\nDay-of-week mean valence:")
    for r in stats["day_of_week"]:
        summary_lines.append(f"  • {r['dow']:<10}  n={r['n']:>2}  valence={r['mean_valence']:+.3f}")

    prompt = _suggest_prompt().format(stats_summary="\n".join(summary_lines))

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return [str(s) for s in data.get("suggestions", [])]
    except (json.JSONDecodeError, KeyError):
        return []


# ============================================================ Plotly figures
def emotion_timeline_fig() -> go.Figure:
    """Daily-mean valence line + positive/negative bands."""
    df = _entries_df()
    if df.empty:
        return go.Figure()

    daily = (df.groupby("date")
               .agg(valence=("valence", "mean"))
               .reset_index())
    daily["date"] = pd.to_datetime(daily["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["valence"],
        mode="lines+markers",
        line=dict(color="#7C3AED", width=2),
        marker=dict(size=7),
        name="Daily mood (valence)",
    ))
    fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Date",
        yaxis_title="Valence (positive − negative emotion)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def topic_bars_fig(stats: dict) -> go.Figure:
    """Side-by-side: top mood-boosters vs top mood-killers."""
    pos = stats.get("topics_pos", [])[:5]
    neg = stats.get("topics_neg", [])[:5]
    fig = go.Figure()
    if pos:
        fig.add_trace(go.Bar(
            y=[r["topic"] for r in pos][::-1],
            x=[r["mean_valence"] for r in pos][::-1],
            orientation="h",
            marker_color="#10B981",
            name="Mood-boosters",
        ))
    if neg:
        fig.add_trace(go.Bar(
            y=[r["topic"] for r in neg][::-1],
            x=[r["mean_valence"] for r in neg][::-1],
            orientation="h",
            marker_color="#EF4444",
            name="Mood-killers",
        ))
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Mean valence when topic is mentioned",
        barmode="overlay",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ============================================================ CLI test
if __name__ == "__main__":
    s = compute_stats()
    print(json.dumps(s, indent=2, default=str))
    if s.get("n_entries", 0) >= 5:
        print("\n--- LLM suggestions ---")
        for line in suggest(s):
            print(" •", line)
