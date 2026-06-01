"""
Hindsight — Ingestion pipeline.

Public API:
    add_entry(text=None, audio_bytes=None, quick_mood=None) -> dict
    rebuild_faiss() -> None
    load_entries() -> list[dict]

What `add_entry` does per call:
  1. If audio_bytes given → Whisper transcribes (in-memory, no disk persistence
     of audio).
  2. Run HF emotion classifier on text → 7-dim emotion score dict.
  3. Run Groq topic-tagger on text → list of topic strings.
  4. Append entry to data/entries.json (with timestamp + uuid).
  5. Embed and add to FAISS index.

Storage format (one entry):
{
  "id": "<uuid>",
  "timestamp": "2026-05-16T18:30:00Z",
  "source": "voice" | "text",
  "text": "...",
  "emotions": {"joy": 0.05, "sadness": 0.6, "anger": 0.1, ...},
  "topics": ["gym", "standup", "mom"],
  "quick_mood": 1..5 | None
}
"""
from __future__ import annotations

import io
import os
import json
import uuid
import pickle
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

# ----------------------------------------------------------------- env / paths
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ROOT       = Path(__file__).parent
DATA       = ROOT / "data"
PROMPTS    = ROOT / "prompts"
DATA.mkdir(exist_ok=True)

ENTRIES_PATH = DATA / "entries.json"
INDEX_PATH   = DATA / "index.faiss"
META_PATH    = DATA / "index_meta.pkl"

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"
WHISPER_MODEL  = "base"


# ============================================================ lazy loaders
@lru_cache(maxsize=1)
def _whisper():
    import whisper
    print("[hindsight] loading Whisper:", WHISPER_MODEL)
    return whisper.load_model(WHISPER_MODEL)


@lru_cache(maxsize=1)
def _emotion_classifier():
    from transformers import pipeline as hf_pipeline
    print("[hindsight] loading emotion classifier:", EMOTION_MODEL)
    return hf_pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        truncation=True,
    )


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    print("[hindsight] loading embedder:", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _groq():
    from groq import Groq
    if not GROQ_API_KEY:
        raise SystemExit("❌ GROQ_API_KEY missing. Copy .env.example to .env.")
    return Groq(api_key=GROQ_API_KEY)


@lru_cache(maxsize=1)
def _topic_prompt() -> str:
    return (PROMPTS / "topic_extraction.txt").read_text()


# ============================================================ helpers
def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def load_entries() -> list[dict]:
    if not ENTRIES_PATH.exists():
        return []
    return json.loads(ENTRIES_PATH.read_text())


def _save_entries(entries: list[dict]) -> None:
    ENTRIES_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


# ============================================================ core steps
def transcribe(audio_bytes: bytes) -> str:
    """Whisper transcription with NO audio persistence."""
    # Whisper wants a file path. We use a NamedTemporaryFile + delete.
    suffix = ".wav"  # st.audio_input returns wav-compatible bytes
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = _whisper().transcribe(tmp.name, fp16=False)
    return result.get("text", "").strip()


def classify_emotions(text: str) -> dict[str, float]:
    """Return {label: probability} for all 7 emotion labels."""
    if not text.strip():
        return {}
    out = _emotion_classifier()(text[:512])   # truncate long inputs
    # HF returns either [[{label, score}, ...]] or [{label, score}, ...] depending on version
    scores = out[0] if isinstance(out[0], list) else out
    return {item["label"].lower(): float(item["score"]) for item in scores}


def extract_topics(text: str) -> list[str]:
    """LLM-driven topic tagger (returns 2-5 normalized tags)."""
    if not text.strip():
        return []
    prompt = _topic_prompt().format(entry_text=text[:3000])
    resp = _groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        topics = data.get("topics", [])
        return [str(t).strip().lower() for t in topics if t][:5]
    except (json.JSONDecodeError, KeyError, AttributeError):
        return []


def embed_text(text: str) -> np.ndarray:
    """Embed → 384-dim float32 unit vector (cosine via inner product)."""
    vec = _embedder().encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32")


# ============================================================ FAISS index
def rebuild_faiss() -> None:
    """Walk all entries, re-embed, write a fresh FAISS index + meta."""
    entries = load_entries()
    if not entries:
        # write an empty index so callers can still call read_index without error
        idx = faiss.IndexFlatIP(384)
        faiss.write_index(idx, str(INDEX_PATH))
        META_PATH.write_bytes(pickle.dumps([]))
        return

    texts = [e["text"] for e in entries]
    vecs  = _embedder().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(INDEX_PATH))
    META_PATH.write_bytes(pickle.dumps([e["id"] for e in entries]))


def load_index() -> tuple[faiss.Index, list[str]]:
    """Return (faiss_index, ordered list of entry_ids matching index rows)."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        rebuild_faiss()
    idx = faiss.read_index(str(INDEX_PATH))
    ids = pickle.loads(META_PATH.read_bytes())
    return idx, ids


# ============================================================ public API
def add_entry(
    *,
    text: str | None = None,
    audio_bytes: bytes | None = None,
    quick_mood: int | None = None,
) -> dict:
    """
    Add one journal entry to the store + index.

    Exactly one of `text` / `audio_bytes` must be supplied.
    `quick_mood` is an optional 1..5 emoji rating.
    """
    if not text and not audio_bytes:
        raise ValueError("Provide either `text` or `audio_bytes`.")

    if audio_bytes:
        source = "voice"
        text   = transcribe(audio_bytes)
    else:
        source = "text"
        text   = text.strip()

    if not text:
        raise ValueError("Empty entry after transcription / cleanup.")

    emotions = classify_emotions(text)
    topics   = extract_topics(text)

    entry = {
        "id":         str(uuid.uuid4())[:12],
        "timestamp":  _now_iso(),
        "source":     source,
        "text":       text,
        "emotions":   emotions,
        "topics":     topics,
        "quick_mood": quick_mood,
    }

    entries = load_entries()
    entries.append(entry)
    _save_entries(entries)
    rebuild_faiss()
    return entry


# ============================================================ CLI test
if __name__ == "__main__":
    print("Adding a text entry…")
    e = add_entry(
        text="Tough day today. Standup went on for 90 minutes and I got "
             "nothing done after. Skipped gym again. Mom called though — "
             "that was the only highlight.",
    )
    print(json.dumps(e, indent=2, ensure_ascii=False))
    print(f"\nTotal entries: {len(load_entries())}")
