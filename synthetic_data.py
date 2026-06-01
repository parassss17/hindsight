"""
Hindsight — Synthetic demo-data generator.

Generates ~30 days of realistic journal entries for a fictional final-year
B.Tech student. Bakes in DETECTABLE PATTERNS so the dashboard + Echo Mode
have something to find:

  • Gym days       → high joy, calm
  • Standup (Tue)  → high anxiety
  • Sundays        → low mood ("Sunday scaries")
  • Mom calls      → mood lift
  • All-nighters   → fatigue + irritation

Each generated text is then run through the SAME pipeline functions used
for real entries — so the resulting `entries.json` is structurally
indistinguishable from real data.

Run once:
    python synthetic_data.py
"""
from __future__ import annotations

import os
import json
import uuid
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# We reuse the pipeline's classifiers + topic extractor + index rebuild.
from pipeline import (
    classify_emotions, extract_topics, rebuild_faiss,
    DATA, ENTRIES_PATH, GROQ_MODEL,
)

load_dotenv()

# ----------------------------------------------------------------- config
N_DAYS         = 30
SEED           = 7
random.seed(SEED)

# Day "themes" — picked randomly, weighted to produce detectable patterns.
THEMES = [
    # (label, weight, mood_hint, content_hint)
    ("gym_high",       3, "energised, optimistic", "did the gym session in the evening, felt great after, productive coding"),
    ("standup_anxiety", 3, "anxious, frustrated",  "long standup meeting in the morning, couldn't focus after, anxious about deadlines"),
    ("sunday_dread",   2, "low, anxious about Monday", "Sunday evening, dreading the week ahead, can't sleep"),
    ("mom_call",       2, "warm, lifted",          "Mom called, talked for an hour about home, felt much better"),
    ("normal_day",     4, "neutral, calm",         "regular day, some coding, some chores, nothing notable"),
    ("all_nighter",    2, "exhausted, irritated",  "stayed up late last night working on the project, running on coffee"),
    ("interview",      1, "nervous then relieved", "had an internship interview, nervous beforehand, went okay"),
    ("friend_meet",    1, "happy, energised",      "met Anmol for coffee, talked about our final-year projects, fun evening"),
]

GENERATION_PROMPT = """You are writing a realistic 60-second voice-journal
entry as if spoken into a phone by a final-year B.Tech (Software Engineering)
student at DTU. The mood is: {mood_hint}. The day involved: {content_hint}.

Rules:
- Write in first person, casual, slightly rambling — like spoken speech
  transcribed.
- 80-180 words.
- Reference SPECIFIC concrete things (gym, standup, Anmol, mom, thesis,
  AlphaCast project, Premortem, sleep, coffee, etc.).
- Do NOT meta-comment. No "today I want to talk about". Just talk.
- Reflect the mood — anxious moods should feel anxious, happy moods light.
- Output ONLY the journal text. No quotes, no preamble.
"""


def _pick_theme():
    """Weighted random choice from THEMES."""
    labels  = [t[0] for t in THEMES]
    weights = [t[1] for t in THEMES]
    chosen  = random.choices(THEMES, weights=weights, k=1)[0]
    return chosen


def _generate_one(mood_hint: str, content_hint: str) -> str:
    """One Groq call → ~120-word journal text."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                mood_hint=mood_hint, content_hint=content_hint),
        }],
        temperature=0.85,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip().strip('"')


def _backdated_timestamp(days_ago: int) -> str:
    base = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    # Add a small random offset for realism (entries written in the evening)
    base = base.replace(hour=21, minute=random.randint(0, 59), second=0, microsecond=0)
    return base.isoformat(timespec="seconds")


def main() -> None:
    print("=" * 60)
    print(f"Hindsight — synthetic data ({N_DAYS} entries)")
    print("=" * 60)
    print(f"NOTE: this overwrites {ENTRIES_PATH} if it exists.\n")

    entries: list[dict] = []
    for i in range(N_DAYS):
        days_ago = N_DAYS - 1 - i               # oldest first
        label, _, mood_hint, content_hint = _pick_theme()

        print(f"[{i+1:>2}/{N_DAYS}] day -{days_ago:>2}  theme={label:<18}", end=" ")

        # 1. Generate the text via Groq
        text = _generate_one(mood_hint, content_hint)
        print(f"({len(text)} chars)")

        # 2. Run real emotion classifier + topic extractor (same as live pipeline)
        emotions = classify_emotions(text)
        topics   = extract_topics(text)

        entries.append({
            "id":         str(uuid.uuid4())[:12],
            "timestamp":  _backdated_timestamp(days_ago),
            "source":     "voice" if random.random() < 0.6 else "text",
            "text":       text,
            "emotions":   emotions,
            "topics":     topics,
            "quick_mood": None,
            "synthetic":  True,    # flag so the UI can show a "demo data" badge
        })

    # 3. Sort chronologically, write to disk, rebuild FAISS
    entries.sort(key=lambda e: e["timestamp"])
    DATA.mkdir(exist_ok=True)
    ENTRIES_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"\n[io] wrote {ENTRIES_PATH} ({len(entries)} entries)")

    print("[faiss] rebuilding index…")
    rebuild_faiss()

    # Quick summary
    print("\n──── Topic frequency ────")
    from collections import Counter
    topic_counts = Counter(t for e in entries for t in e["topics"])
    for tag, n in topic_counts.most_common(12):
        print(f"  {tag:<18}  ×{n}")

    print("\n✅ Demo data ready. Next: build insights.py / app.py.")


if __name__ == "__main__":
    main()
