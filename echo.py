"""
Hindsight — Echo Mode ("Have I Been Here Before?").

Public API:
    find_similar_episodes(feeling_text, k=5) -> list[dict]
    summarize(feeling_text, k=5)            -> dict (LLM-grounded synthesis)

Per call:
  1. Embed current feeling.
  2. FAISS top-k semantically similar past entries.
  3. For each similar entry, look at entries 3-7 days AFTER it ("recovery
     window") to estimate how the episode resolved.
  4. Feed the structured episodes into the echo_summary prompt.
  5. Return a JSON with: n_similar, median_recovery_days, recovery pattern,
     a calm grounded reflection, and one concrete next-action.

Designed to be the "crisis-defuser" feature — receipts that THIS exact
feeling has happened before and resolved.
"""
from __future__ import annotations

import os
import json
import statistics
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pipeline import (
    load_entries, load_index, embed_text, GROQ_MODEL,
)

load_dotenv()
PROMPTS = Path(__file__).parent / "prompts"

RECOVERY_WINDOW_MIN = 1     # days after the similar entry
RECOVERY_WINDOW_MAX = 7
SIMILARITY_FLOOR    = 0.25  # ignore very-loose matches


@lru_cache(maxsize=1)
def _echo_prompt() -> str:
    return (PROMPTS / "echo_summary.txt").read_text()


def _by_id(entries: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in entries}


def _valence(entry: dict) -> float:
    emo = entry.get("emotions", {}) or {}
    pos = emo.get("joy", 0.0) + emo.get("neutral", 0.0)
    neg = (emo.get("sadness", 0.0) + emo.get("anger", 0.0)
           + emo.get("fear", 0.0) + emo.get("disgust", 0.0))
    return float(pos - neg)


def _find_recovery(episode: dict, all_entries: list[dict]) -> tuple[dict | None, int | None]:
    """For one similar episode, find the first entry 1-7 days later whose
    valence is meaningfully higher (>= +0.15 above episode's own valence)."""
    eps_time = pd.to_datetime(episode["timestamp"])
    eps_val  = _valence(episode)
    lo = eps_time + timedelta(days=RECOVERY_WINDOW_MIN)
    hi = eps_time + timedelta(days=RECOVERY_WINDOW_MAX)

    candidates = []
    for e in all_entries:
        t = pd.to_datetime(e["timestamp"])
        if lo <= t <= hi:
            if _valence(e) >= eps_val + 0.15:
                candidates.append((t, e))
    if not candidates:
        return None, None
    candidates.sort()
    recovery_entry = candidates[0][1]
    days = (pd.to_datetime(recovery_entry["timestamp"]) - eps_time).days
    return recovery_entry, days


def find_similar_episodes(feeling_text: str, k: int = 5) -> list[dict]:
    """Top-k similar past episodes + their recovery info."""
    entries = load_entries()
    if not entries:
        return []

    idx, id_list = load_index()
    if idx.ntotal == 0:
        return []

    q = embed_text(feeling_text)
    scores, ids = idx.search(q, min(k * 2, idx.ntotal))

    by_id = _by_id(entries)
    out = []
    seen = set()
    for score, i in zip(scores[0], ids[0]):
        if i == -1 or i >= len(id_list):
            continue
        entry_id = id_list[i]
        if entry_id in seen:
            continue
        seen.add(entry_id)
        if float(score) < SIMILARITY_FLOOR:
            continue
        episode = by_id.get(entry_id)
        if not episode:
            continue
        recovery, days = _find_recovery(episode, entries)
        out.append({
            "episode":  episode,
            "score":    float(score),
            "recovery": recovery,
            "days_to_recovery": days,
        })
        if len(out) >= k:
            break
    return out


def summarize(feeling_text: str, k: int = 5) -> dict:
    """Grounded LLM reflection on the similar episodes."""
    episodes = find_similar_episodes(feeling_text, k=k)
    if not episodes:
        return {
            "n_similar": 0,
            "median_recovery_days": None,
            "recovery_action_pattern": "",
            "reflection": "I don't see any past entries that match how you "
                          "feel right now. That can be a good sign — or it "
                          "can mean we just need more journal data. Either "
                          "way, you're at a fresh starting point.",
            "next_action": "Write a short journal entry about today, even "
                           "just two sentences. Future-you will need it.",
        }

    # Format for the prompt
    formatted = []
    for ep in episodes:
        eps = ep["episode"]
        rec = ep["recovery"]
        formatted.append({
            "episode_date":     str(pd.to_datetime(eps["timestamp"]).date()),
            "episode_text":     eps["text"][:600],
            "recovery_text":    rec["text"][:600] if rec else "(no clear recovery within 7 days)",
            "days_to_recovery": ep["days_to_recovery"],
        })

    prompt = _echo_prompt().format(
        current_feeling=feeling_text.strip()[:1000],
        similar_episodes=json.dumps(formatted, indent=2, ensure_ascii=False),
    )

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    try:
        out = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        out = {}

    # Always overwrite n_similar + median_recovery from real data
    out["n_similar"] = len(episodes)
    days_list = [ep["days_to_recovery"] for ep in episodes if ep["days_to_recovery"]]
    out["median_recovery_days"] = (round(statistics.median(days_list), 1)
                                    if days_list else None)
    out["episodes_detail"] = formatted
    return out


if __name__ == "__main__":
    test_feeling = ("I feel like nothing's working out. I'm so behind on my "
                    "project and everyone seems ahead of me.")
    result = summarize(test_feeling)
    print(json.dumps(result, indent=2, ensure_ascii=False))
