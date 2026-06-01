"""
Hindsight — Talk to Past You.

Public API:
    list_dates_with_entries() -> list[date]
    chat(target_date, user_message, history) -> str

Per call:
  1. Pull all entries within ±3 days of target_date.
  2. Format as bullet list (date + text).
  3. Build a persona prompt grounded ONLY in those entries.
  4. Append conversation history; call Groq.
  5. Return the past-self's reply.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pipeline import load_entries, GROQ_MODEL

load_dotenv()
PROMPTS = Path(__file__).parent / "prompts"

WINDOW_DAYS  = 3        # ± window around target_date
MAX_ENTRIES  = 12       # cap on entries fed into the prompt


@lru_cache(maxsize=1)
def _persona_prompt() -> str:
    return (PROMPTS / "past_persona.txt").read_text()


def list_dates_with_entries() -> list[date]:
    """Sorted unique dates that have at least one entry — for the UI slider."""
    entries = load_entries()
    if not entries:
        return []
    dates = sorted({pd.to_datetime(e["timestamp"]).date() for e in entries})
    return dates


def _entries_near(target: date) -> list[dict]:
    """Entries within ±WINDOW_DAYS of target_date, oldest → newest."""
    entries = load_entries()
    rows = []
    for e in entries:
        d = pd.to_datetime(e["timestamp"]).date()
        if abs((d - target).days) <= WINDOW_DAYS:
            rows.append({"date": d, **e})
    rows.sort(key=lambda r: r["date"])
    return rows[:MAX_ENTRIES]


def _format_entries_for_prompt(entries: list[dict]) -> str:
    out = []
    for r in entries:
        text = r["text"].replace("\n", " ").strip()
        out.append(f"[{r['date']}]  {text[:600]}")
    return "\n".join(out) if out else "(no entries in this window)"


def chat(
    target_date: date,
    user_message: str,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Send one turn of conversation. `history` is a list of
    {role: 'user'|'assistant', content: '...'} dicts from prior turns.
    Returns (assistant_reply, updated_history).
    """
    history = list(history or [])

    entries = _entries_near(target_date)
    if not entries:
        reply = ("I can't remember anything from that period — there are no "
                 "entries within a few days of that date.")
        history += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return reply, history

    # Build the persona-grounded SYSTEM message (re-supplied every turn so
    # the model never drifts off the entries).
    system = _persona_prompt().format(
        entries=_format_entries_for_prompt(entries),
        user_message=user_message,
    )

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # We use the SYSTEM message as the persona-and-context block, then
    # append prior conversation history + the new user turn.
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.6,
        max_tokens=240,
    )
    reply = resp.choices[0].message.content.strip()

    history += [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return reply, history


if __name__ == "__main__":
    dates = list_dates_with_entries()
    if not dates:
        print("No entries — run `python synthetic_data.py` first.")
        raise SystemExit

    target = dates[len(dates) // 2]
    print(f"Talking to past-you from {target}\n")
    reply, hist = chat(target, "Hey past me, what was bothering you?")
    print("[Past You]", reply, "\n")

    reply, hist = chat(target, "Did it get better?", history=hist)
    print("[Past You]", reply)
