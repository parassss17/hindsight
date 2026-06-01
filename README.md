---
title: Hindsight
emoji: 🪞
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🪞 Hindsight — AI Companion for Voice-Journal Reflection

> Record a 60-second voice rant (or type) every day. Over weeks, Hindsight
> learns what makes you happy, what stresses you, and surfaces actionable
> patterns grounded in *your own* past entries.

**Live demo:** _(Hugging Face Spaces URL goes here after deploy)_
**Source:** _(GitHub URL goes here)_

---

## Why this exists

Most journaling apps store entries. Hindsight *reflects them back to you*.
After ~30 days of data it:

1. **Plots your emotional timeline** across 6 dimensions (joy, sadness,
   anxiety, anger, calm, fatigue).
2. **Cross-references emotions × topics × time** to identify causal links
   ("you were happiest on gym days, most anxious on standup Tuesdays").
3. **Generates actionable suggestions** grounded entirely in your own data.

Plus two signature features:
- 🪞 **Talk to Past You** — slide to any past date and chat with a
  simulated version of yourself from that period, grounded in entries
  from that window.
- 🔁 **Echo Mode** — in a crisis moment, retrieves the most-similar past
  episodes, computes recovery timelines, and surfaces what worked.

---

## Architecture

```
USER INPUT
  ├── 🎤 in-browser voice recording (st.audio_input)
  └── ⌨ text entry
       │
       ▼
   audio? → [ Whisper base — in-memory transcription ]
       │
       ▼
   text entry
       │
       ▼
[ HF emotion classifier ] → 6-dim emotion scores
       │
       ▼
[ Groq Llama-3 — topic tagger ] → ["gym", "standup", "mom", …]
       │
       ▼
[ embed via MiniLM → store in FAISS ] + entries.json append
       │
       ▼
THREE LIVE EXPERIENCES (Streamlit tabs):
  ├── Today: capture entries
  ├── Dashboard: emotion timeline + suggestions
  └── Memory: Talk to Past You + Echo Mode
```

---

## Privacy by design

- 🛡 **Audio is transcribed in-memory and immediately discarded.**
- 🛡 Only the resulting text persists, locally in `data/entries.json`.
- 🛡 Nothing is sent to external services except the Groq LLM call (which
  receives only your text, never audio).
- 🛡 Personal entries are gitignored. The public demo uses synthetic
  LLM-generated data.

---

## Quickstart (local)

```bash
git clone <repo-url>
cd hindsight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # then fill in GROQ_API_KEY
python synthetic_data.py              # generates 30 days of demo entries (~1 min)
streamlit run app.py                  # opens at http://localhost:8501
```

---

## Tech stack

`Python` · `OpenAI Whisper (base)` · `HuggingFace Transformers
(j-hartmann/emotion-english-distilroberta-base)` · `sentence-transformers
(MiniLM)` · `FAISS` · `Groq (Llama-3.3-70B)` · `Streamlit` · `Plotly` ·
`Hugging Face Spaces`

---

## Author

**Paras** — B.Tech (Software Engineering), Delhi Technological University.
Sibling projects:
- [**AlphaCast**](https://huggingface.co/spaces/parassss17/alphacast) — sector-augmented Transformer vs LSTM for stock forecasting
- [**Premortem**](https://huggingface.co/spaces/parassss17/premortem) — inverse-RAG over 590 documented project failures
