# app.py — minimal: Record -> Prompt -> Transcribe & Summarize -> Download ZIP
import os
import io
import re
import tempfile
import zipfile
from datetime import datetime

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
import av
import numpy as np
from pydub import AudioSegment

# ---- faster-whisper (local STT) ----
try:
    from faster_whisper import WhisperModel
    HAVE_FASTER = True
except Exception:
    HAVE_FASTER = False

# ---- Gemini (minutes) from secrets/env ----
import google.generativeai as genai

@st.cache_resource
def configure_gemini() -> bool:
    key = (getattr(st, "secrets", {}).get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)
    key = key or os.getenv("GEMINI_API_KEY")
    if not key:
        return False
    genai.configure(api_key=key)
    return True

gemini_ready = configure_gemini()

# ---- cache one whisper model for all users ----
@st.cache_resource(show_spinner="Loading faster-whisper model...")
def get_whisper(model_size: str = "base", device: str = "auto", compute_type: str = "int8") -> WhisperModel:
    dev = device if device in {"cpu", "cuda"} else "auto"
    return WhisperModel(model_size, device=dev, compute_type=compute_type)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Meeting Minutes Generator", page_icon="📝", layout="wide")
st.title("📝 AI Meeting Minutes Generator")

# Sidebar (small status only)
with st.sidebar:
    st.caption("🔊 Recording uses WebRTC. If mic is blocked, open in a browser tab and allow microphone.")
    st.caption("🧠 STT: faster-whisper (local) • Minutes: Gemini")
    st.caption("✅ Gemini configured from server secrets." if gemini_ready else "⚠️ Gemini key missing (set .streamlit/secrets.toml or GEMINI_API_KEY).")

# ---- session state ----
if "full_audio" not in st.session_state:
    st.session_state.full_audio = AudioSegment.silent(duration=0)
if "final_transcript" not in st.session_state:
    st.session_state.final_transcript = ""
if "minutes" not in st.session_state:
    st.session_state.minutes = ""
if "webrtc_key" not in st.session_state:
    st.session_state.webrtc_key = 0
if "mic_denied" not in st.session_state:
    st.session_state.mic_denied = False
if "webrtc_last_error" not in st.session_state:
    st.session_state.webrtc_last_error = ""

# ---- helpers ----
def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def np_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = float_to_pcm16(audio_np)
    seg = AudioSegment(data=pcm16.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()

def save_chunk_to_session(wav_bytes: bytes):
    new = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    st.session_state.full_audio += new

def export_full_audio_wav() -> bytes:
    buf = io.BytesIO()
    st.session_state.full_audio.export(buf, format="wav")
    return buf.getvalue()

def transcribe_audiosegment_with_fw(seg: AudioSegment, language: str | None = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        seg.export(tf.name, format="wav")
        model = get_whisper("base", "auto", "int8")
        segments, _ = model.transcribe(tf.name, language=language, vad_filter=True)
        return " ".join(s.text for s in segments).strip()

def generate_minutes_with_gemini(transcript: str, user_prompt: str) -> str:
    if not gemini_ready:
        raise RuntimeError("Gemini is not configured on the server.")
    base_prompt = f"""
You are an expert meeting minutes writer.

User instruction:
{user_prompt}

Use the transcript below to produce concise, business-ready minutes.
- Bullet points where appropriate.
- Capture key points, decisions, action items (owners/dates if present), and a brief summary.
- If the user asked for a different structure, follow it.

Transcript:
\"\"\"{transcript}\"\"\"
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(base_prompt)
    return (getattr(resp, "text", "") or "").strip()

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:60] or "meeting"

# ---- mic help / error capture ----
def _render_mic_help():
    with st.expander("How to enable microphone access", expanded=False):
        st.markdown(
            """
**Chrome / Edge**
1) Click the **🔒 lock** in the address bar  
2) **Site settings** → **Microphone** → **Allow**  
3) Reload this page

**Firefox**
1) Click the **mic icon** in the address bar  
2) Choose **Allow** (optionally “Remember this decision”)  
3) Reload this page

**Safari (macOS)**
1) **Safari → Settings → Websites → Microphone**  
2) Set this site to **Allow**  
3) Reload this page
"""
        )
        if hasattr(st, "rerun") and st.button("Reload now"):
            st.rerun()

class AudioChunker(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = 16000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert frame to numpy
        pcm = frame.to_ndarray()

        # Convert to mono if stereo/multi-channel
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        # Normalize to float32 between -1 and 1
        pcm = pcm.astype(np.float32)
        pcm /= np.iinfo(frame.to_ndarray().dtype).max

        # Save chunk to session
        wav_bytes = np_to_wav_bytes(pcm, self.sample_rate)
        save_chunk_to_session(wav_bytes)

        return frame

# ---- Main UI (single page) ----
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("1) Record")
    rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key=f"rec-{st.session_state.webrtc_key}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioChunker,
    )
    # Retry button: rebuilds the widget so the browser can re-request mic
    if st.button("Retry mic setup"):
        st.session_state.webrtc_key += 1
        if hasattr(st, "rerun"):
            st.rerun()

    # Auto banner when permission blocked or widget not playing
     # Show banner when the widget exists but isn't playing (common after a prior Deny)
    if ctx and not ctx.state.playing:
        st.info("No Audio Yet")
        with st.expander("How to enable microphone access", expanded=False):
            st.markdown(
                """
**Chrome / Edge**
1) Click the **🔒 lock** in the address bar  
2) **Site settings** → **Microphone** → **Allow**  
3) Reload this page

**Firefox**
1) Click the **mic icon** in the address bar  
2) Choose **Allow** (optionally “Remember this decision”)  
3) Reload this page

**Safari (macOS)**
1) **Safari → Settings → Websites → Microphone**  
2) Set this site to **Allow**  
3) Reload this page
                """
            )
            if hasattr(st, "rerun") and st.button("Reload now"):
                st.rerun()

    st.markdown("When you click **Start**, we record your mic until you click **Stop**.")

with col_right:
    st.subheader("2) Title & Prompt")
    title = st.text_input("Title (used for filenames)", placeholder="Weekly Sales Sync", value="")
    user_prompt = st.text_area(
        "Prompt for AI (what to do with the audio)",
        value="Create clean minutes: key points, action details, and a short summary.",
        height=120,
    )
    language_hint = st.text_input("Language hint (optional, e.g., en, ms, zh)", value="")

st.divider()

c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("📝 Transcribe & Generate"):
        if len(st.session_state.full_audio) == 0:
            st.warning("No audio recorded yet.")
        else:
            with st.spinner("Transcribing (faster-whisper)..."):
                st.session_state.final_transcript = transcribe_audiosegment_with_fw(
                    st.session_state.full_audio, language=language_hint or None
                )
            if not gemini_ready:
                st.error("Gemini is not configured (set .streamlit/secrets.toml or GEMINI_API_KEY).")
            else:
                with st.spinner("Generating minutes (Gemini)..."):
                    st.session_state.minutes = generate_minutes_with_gemini(
                        st.session_state.final_transcript, user_prompt
                    )
            st.success("Done. Scroll down to preview & download.")

with c2:
    if st.button("🧹 Clear recording"):
        st.session_state.full_audio = AudioSegment.silent(duration=0)
        st.session_state.final_transcript = ""
        st.session_state.minutes = ""
        st.success("Cleared.")

with c3:
    if st.button("💾 Download audio.wav"):
        wav_bytes = export_full_audio_wav()
        st.download_button("Save audio.wav", data=wav_bytes, file_name="audio.wav", mime="audio/wav")

# Preview
st.subheader("Preview")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Transcript**")
    st.text_area("transcript.txt", st.session_state.final_transcript, height=260)
with col_b:
    st.markdown("**Minutes**")
    st.text_area("minutes.txt", st.session_state.minutes, height=260)

# ---- Package download (ZIP) ----
st.subheader("3) Download package (audio + transcript + minutes)")

def build_zip_package(title_str: str) -> bytes:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{slugify(title_str) or 'meeting'}_{ts}"
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}/audio.wav", export_full_audio_wav())
        zf.writestr(f"{base}/transcript.txt", (st.session_state.final_transcript or "").encode("utf-8"))
        zf.writestr(f"{base}/minutes.txt", (st.session_state.minutes or "").encode("utf-8"))
        meta = f"Title: {title_str or 'meeting'}\nCreated: {ts}\n"
        zf.writestr(f"{base}/meta.txt", meta.encode("utf-8"))
    mem.seek(0)
    return mem.getvalue()

pkg_disabled = (len(st.session_state.full_audio) == 0)
pkg_bytes = build_zip_package(title) if not pkg_disabled else None

st.download_button(
    "⬇️ Download ZIP",
    data=pkg_bytes if pkg_bytes else b"",
    file_name=f"{slugify(title) or 'meeting'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
    mime="application/zip",
    disabled=pkg_disabled,
)
