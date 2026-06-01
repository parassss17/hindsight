"""
Hindsight — Streamlit UI.

Three tabs:
  1. Today    — record voice / type / quick-mood → add entry
  2. Dashboard — emotion timeline + topic patterns + LLM suggestions
  3. Memory   — Talk to Past You (date-windowed chat) + Echo Mode

Privacy note shown prominently on Today tab.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

import pipeline
import insights
import past_chat
import echo

# ============================================================ page config
st.set_page_config(
    page_title="Hindsight — Voice Journal AI",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================ helpers
def _is_demo_data() -> bool:
    entries = pipeline.load_entries()
    return bool(entries) and any(e.get("synthetic") for e in entries)


def _count_entries() -> int:
    return len(pipeline.load_entries())


# ============================================================ header
st.title("🪞 Hindsight")
st.caption("Voice-journal AI that learns YOUR emotional patterns over time.")

n = _count_entries()
if n == 0:
    st.warning(
        "No entries yet. Run `python synthetic_data.py` for a 30-day demo "
        "corpus, or just start writing in the **Today** tab."
    )
elif _is_demo_data():
    st.info(
        f"Showing {n} demo entries (LLM-generated for a fictional "
        f"final-year B.Tech student). Real users would record their own."
    )

tab_today, tab_dash, tab_mem = st.tabs(
    ["📝 Today", "📊 Dashboard", "🪞 Memory"]
)


# ============================================================ TAB 1: Today
with tab_today:
    st.subheader("How was today?")
    st.markdown(
        "<small>🛡 Audio is transcribed in-memory and immediately discarded — "
        "only the resulting text is stored.</small>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🎤 Record (in-browser)**")

        audio_bytes: bytes | None = None

        # --- preferred path: streamlit-mic-recorder (works on any Streamlit) ---
        try:
            from streamlit_mic_recorder import mic_recorder
            rec = mic_recorder(
                start_prompt="🎤 Tap to record",
                stop_prompt="⏹ Stop recording",
                just_once=False,        # tap again to re-record
                use_container_width=True,
                format="wav",
                key="mic",
            )
            if rec and rec.get("bytes"):
                audio_bytes = rec["bytes"]
        except Exception:
            # --- fallback 1: native st.audio_input (Streamlit >= 1.31) ---
            if hasattr(st, "audio_input"):
                audio = st.audio_input("Tap to record your day", key="rec_native")
                if audio is not None:
                    audio_bytes = audio.getvalue()
            # --- fallback 2: file uploader (works everywhere) ---
            else:
                up = st.file_uploader(
                    "Upload an audio file (wav / mp3 / m4a / ogg)",
                    type=["wav", "mp3", "m4a", "ogg", "webm"],
                    key="rec_upload",
                )
                if up is not None:
                    audio_bytes = up.getvalue()

        # 🔊 Playback — let the user actually LISTEN to what they recorded
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.caption("🔁 Don't like it? Just tap the record button again to redo.")

        rec_save = st.button("Save voice entry", type="primary",
                              use_container_width=True,
                              disabled=audio_bytes is None)
        if rec_save and audio_bytes:
            with st.spinner("Transcribing + tagging…"):
                entry = pipeline.add_entry(audio_bytes=audio_bytes)
            st.success(f"Saved entry {entry['id']}.")
            st.write("**Transcript:**", entry["text"])
            st.write("**Topics:**", ", ".join(entry["topics"]) or "—")

    with col_b:
        st.markdown("**⌨ Type instead**")
        txt = st.text_area("How was today?",
                            height=180, label_visibility="collapsed",
                            placeholder="A few sentences about your day…")
        mood = st.radio("Quick mood (optional)",
                         options=[1, 2, 3, 4, 5],
                         format_func=lambda x: {1: "😢", 2: "😟", 3: "😐",
                                                 4: "🙂", 5: "😊"}[x],
                         horizontal=True, index=2, key="mood")
        txt_save = st.button("Save text entry", type="primary",
                              use_container_width=True,
                              disabled=not txt.strip())
        if txt_save and txt.strip():
            with st.spinner("Tagging…"):
                entry = pipeline.add_entry(text=txt, quick_mood=mood)
            st.success(f"Saved entry {entry['id']}.")
            st.write("**Topics:**", ", ".join(entry["topics"]) or "—")


# ============================================================ TAB 2: Dashboard
with tab_dash:
    if n == 0:
        st.info("Add some entries first.")
    else:
        with st.spinner("Computing patterns…"):
            stats = insights.compute_stats()

        c1, c2, c3 = st.columns(3)
        c1.metric("Entries", stats["n_entries"])
        c2.metric("Avg valence",
                   f"{stats['avg_valence']:+.2f}",
                   help="Positive minus negative emotion, range ~[-1, +1]")
        if stats["trend"]["prev_7"] is not None:
            delta = stats["trend"]["last_7"] - stats["trend"]["prev_7"]
            c3.metric("Last-7 vs prev-7",
                       f"{stats['trend']['last_7']:+.2f}",
                       delta=f"{delta:+.2f}")

        st.markdown("---")
        st.markdown("### Emotional timeline")
        st.plotly_chart(insights.emotion_timeline_fig(), use_container_width=True)

        st.markdown("### What moves your mood")
        st.caption("Topics shown only if mentioned ≥ 2 times.")
        st.plotly_chart(insights.topic_bars_fig(stats), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Actionable patterns")
        st.caption("Generated by an LLM grounded entirely in your stats above.")
        with st.spinner("Synthesising suggestions…"):
            for s in insights.suggest(stats):
                st.markdown(f"- {s}")


# ============================================================ TAB 3: Memory
with tab_mem:
    if n == 0:
        st.info("Add some entries first.")
    else:
        mem_left, mem_right = st.columns(2)

        # ────────────────── Talk to Past You ──────────────────
        with mem_left:
            st.markdown("### 🪞 Talk to Past You")
            st.caption("Chat with a simulated version of yourself from a past date. "
                        "All responses grounded only in entries from a ±3-day window.")

            dates = past_chat.list_dates_with_entries()
            if not dates:
                st.info("Need at least one entry.")
            else:
                target = st.select_slider(
                    "Pick a date",
                    options=dates,
                    value=dates[len(dates) // 2],
                    format_func=lambda d: d.strftime("%a %b %d, %Y"),
                )

                # Reset chat if user changes the date
                if st.session_state.get("past_target") != target:
                    st.session_state["past_target"] = target
                    st.session_state["past_history"] = []

                for msg in st.session_state.get("past_history", []):
                    role = msg["role"]
                    if role == "user":
                        st.markdown(f"**You (present):** {msg['content']}")
                    else:
                        st.markdown(f"**Past You ({target.strftime('%b %d')}):** {msg['content']}")

                user_msg = st.chat_input("Ask past-you something…")
                if user_msg:
                    with st.spinner("Past you is replying…"):
                        reply, hist = past_chat.chat(
                            target_date=target,
                            user_message=user_msg,
                            history=st.session_state.get("past_history", []),
                        )
                    st.session_state["past_history"] = hist
                    st.rerun()

        # ────────────────── Echo Mode ──────────────────
        with mem_right:
            st.markdown("### 🔁 Have I been here before?")
            st.caption("Describe what you're feeling right now. Hindsight finds "
                        "the closest past episodes and shows how each resolved.")

            feeling = st.text_area(
                "Current feeling",
                height=120,
                placeholder="E.g. 'I feel like nothing's working. I'm so behind.'",
                key="echo_input",
                label_visibility="collapsed",
            )
            go = st.button("🔁 Find similar episodes", use_container_width=True,
                            disabled=not feeling.strip())

            if go and feeling.strip():
                with st.spinner("Searching your past…"):
                    result = echo.summarize(feeling)

                if result.get("n_similar", 0) == 0:
                    st.info(result.get("reflection",
                                        "No similar past episodes found."))
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("Similar episodes", result["n_similar"])
                    med = result.get("median_recovery_days")
                    c2.metric("Median recovery",
                               f"{med} days" if med is not None else "—")

                    st.markdown(f"**Pattern:** {result.get('recovery_action_pattern', '—')}")
                    st.markdown(f"> {result.get('reflection', '')}")
                    st.success(f"🎯 Next action: {result.get('next_action', '')}")

                    with st.expander(f"📂 The {result['n_similar']} similar episodes"):
                        for ep in result.get("episodes_detail", []):
                            st.markdown(
                                f"**{ep['episode_date']}**  "
                                f"<small>(recovered in "
                                f"{ep['days_to_recovery']} days)</small>" if ep["days_to_recovery"]
                                else f"**{ep['episode_date']}**  <small>(no clear recovery)</small>",
                                unsafe_allow_html=True,
                            )
                            st.caption("**Episode:** " + ep["episode_text"])
                            st.caption("**Recovery:** " + ep["recovery_text"])
                            st.markdown("---")


# ============================================================ footer
st.divider()
st.markdown(
    "<center><small>Built by <b>Paras</b> · "
    "<a href='https://github.com/parassss17/hindsight'>source</a> · "
    "sibling projects: "
    "<a href='https://huggingface.co/spaces/parassss17/alphacast'>AlphaCast</a>"
    " · "
    "<a href='https://huggingface.co/spaces/parassss17/premortem'>Premortem</a>"
    "</small></center>",
    unsafe_allow_html=True,
)
