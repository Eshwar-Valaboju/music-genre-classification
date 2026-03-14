import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib
import requests
import re
import tempfile
import os
import subprocess

# ─────────────────────────────────────────────
#  Spotify credentials
# ─────────────────────────────────────────────
SPOTIPY_CLIENT_ID     = "93de3f5d760346a4b23415e675bd2bfb"
SPOTIPY_CLIENT_SECRET = "560850a0ad6a4343aafdd19a01d8f12e"

CLIP_DURATION = 30
MAX_CLIPS     = 10

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SoundPrint — Genre Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --bg:          #07070f;
  --surface:     #0f0f1a;
  --surface2:    #16162a;
  --border:      rgba(139,92,246,0.18);
  --border-glow: rgba(139,92,246,0.45);
  --accent1:     #8B5CF6;
  --accent2:     #06B6D4;
  --accent3:     #F472B6;
  --text:        #e8e8f0;
  --muted:       #6b6b8a;
  --success:     #10b981;
  --warning:     #f59e0b;
  --error:       #ef4444;
  --radius:      16px;
  --radius-sm:   10px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

.stApp {
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(139,92,246,0.15) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 100%, rgba(6,182,212,0.08) 0%, transparent 60%),
    url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 2rem 1.5rem 4rem !important;
  max-width: 780px !important;
}

h1, h2, h3, h4 {
  font-family: 'Syne', sans-serif !important;
  letter-spacing: -0.02em !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 100px !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
  width: fit-content !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 100px !important;
  padding: 8px 24px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  transition: all 0.25s ease !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
  box-shadow: 0 4px 20px rgba(139,92,246,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.stTextInput > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div:focus-within {
  border-color: var(--accent1) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.18), 0 0 20px rgba(139,92,246,0.12) !important;
}
.stTextInput input {
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  background: transparent !important;
}
.stTextInput label {
  color: var(--muted) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent1) 0%, #7c3aed 50%, var(--accent2) 100%) !important;
  background-size: 200% 200% !important;
  animation: gradientShift 4s ease infinite !important;
  color: white !important;
  border: none !important;
  border-radius: 100px !important;
  padding: 0.65rem 2.2rem !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  cursor: pointer !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  box-shadow: 0 4px 24px rgba(139,92,246,0.35) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(139,92,246,0.55) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
  border-radius: 100px !important;
}
.stProgress > div > div {
  background: var(--surface2) !important;
  border-radius: 100px !important;
  height: 6px !important;
}

div[data-testid="stInfoMessage"] {
  background: rgba(6,182,212,0.08) !important;
  border: 1px solid rgba(6,182,212,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent2) !important;
}
div[data-testid="stSuccessMessage"] {
  background: rgba(16,185,129,0.08) !important;
  border: 1px solid rgba(16,185,129,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--success) !important;
}
div[data-testid="stWarningMessage"] {
  background: rgba(245,158,11,0.08) !important;
  border: 1px solid rgba(245,158,11,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--warning) !important;
}
div[data-testid="stErrorMessage"] {
  background: rgba(239,68,68,0.08) !important;
  border: 1px solid rgba(239,68,68,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--error) !important;
}

[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.5rem !important;
  transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent1) !important;
  background: rgba(139,92,246,0.05) !important;
}

audio {
  width: 100% !important;
  border-radius: var(--radius-sm) !important;
  filter: invert(1) hue-rotate(180deg) brightness(0.85) !important;
  margin: 0.5rem 0 !important;
}

.streamlit-expanderHeader {
  background: var(--surface) !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

.stSpinner > div { border-top-color: var(--accent1) !important; }
[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.85rem !important; }

hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent) !important;
  margin: 1.5rem 0 !important;
}

[data-testid="stImage"] img {
  border-radius: var(--radius) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}

/* ── Custom components ── */
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.3);
  border-radius: 100px; padding: 5px 14px;
  font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent1); margin-bottom: 0.8rem;
  font-family: 'DM Sans', sans-serif;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.2rem, 5vw, 3.2rem);
  font-weight: 800; line-height: 1.1;
  background: linear-gradient(135deg, #fff 30%, var(--accent1) 70%, var(--accent2) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin: 0 0 0.5rem;
}
.hero-sub {
  color: var(--muted); font-size: 1rem; font-weight: 300;
  line-height: 1.65; margin-bottom: 1.6rem; max-width: 520px;
}
.waveform-deco {
  display: flex; align-items: flex-end;
  gap: 3px; height: 28px; margin: 0.3rem 0 1.8rem;
}
.waveform-deco span {
  display: block; width: 3px; border-radius: 3px;
  background: linear-gradient(to top, var(--accent1), var(--accent2));
  animation: wave 1.4s ease-in-out infinite; opacity: 0.55;
}
.waveform-deco span:nth-child(1)  { height:40%; animation-delay:0.0s; }
.waveform-deco span:nth-child(2)  { height:70%; animation-delay:0.1s; }
.waveform-deco span:nth-child(3)  { height:55%; animation-delay:0.2s; }
.waveform-deco span:nth-child(4)  { height:90%; animation-delay:0.3s; }
.waveform-deco span:nth-child(5)  { height:65%; animation-delay:0.4s; }
.waveform-deco span:nth-child(6)  { height:80%; animation-delay:0.5s; }
.waveform-deco span:nth-child(7)  { height:50%; animation-delay:0.6s; }
.waveform-deco span:nth-child(8)  { height:75%; animation-delay:0.7s; }
.waveform-deco span:nth-child(9)  { height:40%; animation-delay:0.8s; }
.waveform-deco span:nth-child(10) { height:60%; animation-delay:0.9s; }
@keyframes wave {
  0%, 100% { transform: scaleY(1);   opacity: 0.55; }
  50%       { transform: scaleY(0.35); opacity: 0.25; }
}

.track-card {
  background: linear-gradient(135deg, var(--surface) 0%, rgba(139,92,246,0.06) 100%);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.2rem 1.4rem; margin: 1rem 0;
  display: flex; gap: 1.1rem; align-items: flex-start;
  animation: fadeSlideUp 0.4s ease both;
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.track-meta h3 {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.15rem !important; font-weight: 700 !important;
  margin: 0 0 4px !important; color: #fff !important;
}
.track-meta p { font-size: 0.85rem; color: var(--muted); margin: 3px 0; }
.track-meta a {
  color: var(--accent1) !important; text-decoration: none;
  font-size: 0.82rem; font-weight: 500; display: inline-block; margin-top: 6px;
}
.track-meta a:hover { text-decoration: underline; }

.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0; }
.stat-pill {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 10px;
  font-size: 0.75rem; color: var(--muted); font-family: 'DM Sans', sans-serif;
}
.stat-pill strong { color: var(--text); font-weight: 500; }

.result-label {
  font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); margin-bottom: 8px;
}
.result-pill {
  display: inline-flex; align-items: center; gap: 10px;
  border: 1px solid var(--border-glow); border-radius: 100px;
  padding: 12px 32px;
  font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700;
  color: white;
  box-shadow: 0 0 40px rgba(139,92,246,0.2), inset 0 1px 0 rgba(255,255,255,0.06);
  margin: 0.3rem 0 1.2rem;
  animation: resultPop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) both;
}
@keyframes resultPop {
  0%   { transform: scale(0.85); opacity: 0; }
  100% { transform: scale(1);    opacity: 1; }
}

.genre-row {
  display: flex; align-items: center; gap: 10px;
  margin: 6px 0; font-family: 'DM Sans', sans-serif; font-size: 0.86rem;
}
.genre-label { width: 100px; color: var(--text); font-weight: 500; flex-shrink: 0; }
.genre-bar-bg {
  flex: 1; height: 6px; background: var(--surface2);
  border-radius: 100px; overflow: hidden;
}
.genre-bar-fill {
  height: 100%; border-radius: 100px;
  animation: barGrow 0.9s cubic-bezier(0.25,0.46,0.45,0.94) both;
}
@keyframes barGrow { from { width: 0 !important; } }
.genre-pct { width: 38px; text-align: right; color: var(--muted); font-size: 0.78rem; flex-shrink: 0; }

.clip-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 3px 9px;
  font-size: 0.74rem; color: var(--muted); font-family: 'DM Sans', sans-serif; margin: 3px 2px;
}
.clip-badge-genre { color: var(--accent1); font-weight: 600; }

.sp-footer {
  text-align: center; color: var(--muted); font-size: 0.75rem;
  margin-top: 3rem; padding-top: 1.5rem;
  border-top: 1px solid var(--border); font-family: 'DM Sans', sans-serif;
}
.sp-footer span { color: var(--accent1); }

.sp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.2rem 1.4rem; margin: 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Spotify helpers
# ─────────────────────────────────────────────
def get_spotify_token(client_id, client_secret):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret), timeout=10,
    )
    return r.json().get("access_token") if r.status_code == 200 else None

def extract_track_id(url):
    m = re.search(r"spotify\.com/track/([A-Za-z0-9]+)", url)
    return m.group(1) if m else None

def get_track_info(track_id, token):
    r = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {token}"}, timeout=10,
    )
    return r.json() if r.status_code == 200 else None

def download_preview(preview_url):
    try:
        r = requests.get(preview_url, timeout=20); r.raise_for_status()
        return r.content
    except Exception:
        return None

# ─────────────────────────────────────────────
#  yt-dlp / ffmpeg helpers
# ─────────────────────────────────────────────
def check_ytdlp_ffmpeg():
    def ok(cmd):
        try:
            subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
            return True
        except Exception:
            return False
    return ok("yt-dlp"), ok("ffmpeg")

def download_full_youtube(track_name, artist):
    query    = f"{track_name} {artist} official audio"
    tmp_dir  = tempfile.mkdtemp()
    out_tmpl = os.path.join(tmp_dir, "full_audio.%(ext)s")
    cmd = ["yt-dlp", f"ytsearch1:{query}", "--extract-audio",
           "--audio-format", "mp3", "--audio-quality", "5",
           "--no-playlist", "--output", out_tmpl, "--quiet", "--no-warnings"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            return None
        for f in os.listdir(tmp_dir):
            if f.startswith("full_audio"):
                return os.path.join(tmp_dir, f)
        return None
    except Exception:
        return None

def split_audio_into_clips(audio_path, clip_duration=30):
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    try:
        p = subprocess.run(probe, capture_output=True, text=True, timeout=15)
        total = float(p.stdout.strip())
    except Exception:
        total = 300.0
    n_clips = min(int(total // clip_duration), MAX_CLIPS) or 1
    tmp_dir = tempfile.mkdtemp()
    clips   = []
    for i in range(n_clips):
        out = os.path.join(tmp_dir, f"clip_{i:02d}.mp3")
        cmd = ["ffmpeg", "-y", "-ss", str(i * clip_duration),
               "-i", audio_path, "-t", str(clip_duration),
               "-acodec", "libmp3lame", "-q:a", "5", out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and os.path.exists(out):
            clips.append(out)
    return clips

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    try:
        @tf.keras.utils.register_keras_serializable()
        class CustomDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)
        with tf.keras.utils.custom_object_scope({"Dense": CustomDense}):
            model = tf.keras.models.load_model("music_genre_cnn.h5", compile=False)
        scaler = joblib.load("scaler.joblib")
        genre_mapping = {
            0:"rock", 1:"blues", 2:"pop",     3:"metal",
            4:"country", 5:"reggae", 6:"disco", 7:"hiphop",
            8:"jazz",  9:"classical",
        }
        return model, scaler, genre_mapping
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}"); st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}"); st.stop()

# ─────────────────────────────────────────────
#  Feature extraction & prediction
# ─────────────────────────────────────────────
def extract_features(audio_source, sr=22050, n_mfcc=13, n_chroma=12):
    try:
        y, sr = librosa.load(audio_source, sr=sr, duration=CLIP_DURATION)
        if len(y) == 0: return None
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma    = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr       = librosa.feature.zero_crossing_rate(y)
        return np.concatenate([np.mean(mfccs, axis=1), np.mean(chroma, axis=1),
                                [np.mean(spec_cent)], [np.mean(spec_roll)], [np.mean(zcr)]])
    except Exception:
        return None

def predict_single_clip(audio_source):
    model, scaler, _ = load_model_and_scaler()
    feats = extract_features(audio_source)
    if feats is None: return None
    try:
        scaled = scaler.transform(feats.reshape(1, -1))
        return model.predict(np.expand_dims(scaled, axis=-1), verbose=0)[0]
    except Exception:
        return None

# ─────────────────────────────────────────────
#  UI constants
# ─────────────────────────────────────────────
GENRE_EMOJI = {
    "rock":"🎸","blues":"🎷","pop":"🎤","metal":"🤘",
    "country":"🤠","reggae":"🌴","disco":"🕺","hiphop":"🎧",
    "jazz":"🎺","classical":"🎻",
}
GENRE_COLOR = {
    "rock":"#ef4444","blues":"#3b82f6","pop":"#ec4899","metal":"#94a3b8",
    "country":"#f59e0b","reggae":"#10b981","disco":"#8b5cf6","hiphop":"#f97316",
    "jazz":"#06b6d4","classical":"#a78bfa",
}

# ─────────────────────────────────────────────
#  Render helpers
# ─────────────────────────────────────────────
def render_genre_bars(prob_dict):
    html = '<div style="margin:0.4rem 0 1rem;">'
    for genre, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        pct   = prob * 100
        color = GENRE_COLOR.get(genre, "#8b5cf6")
        emoji = GENRE_EMOJI.get(genre, "🎵")
        html += f"""
        <div class="genre-row">
          <div class="genre-label">{emoji} {genre.capitalize()}</div>
          <div class="genre-bar-bg">
            <div class="genre-bar-fill"
                 style="width:{pct:.1f}%;background:linear-gradient(90deg,{color},{color}99);"></div>
          </div>
          <div class="genre-pct">{pct:.0f}%</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_result(genre, prob_dict, n_clips):
    emoji = GENRE_EMOJI.get(genre, "🎵")
    color = GENRE_COLOR.get(genre, "#8b5cf6")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<p class="result-label">🎯 Final Prediction — '
        f'averaged across {n_clips} clip{"s" if n_clips>1 else ""}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="result-pill" style="border-color:{color}66;'
        f'background:linear-gradient(135deg,{color}22,{color}0d);">'
        f'{emoji}&nbsp;{genre.capitalize()}</div>',
        unsafe_allow_html=True,
    )
    render_genre_bars(prob_dict)

def render_per_clip(per_clip):
    if not per_clip: return
    badges = ""
    for i, cp in enumerate(per_clip):
        top   = max(cp, key=cp.get)
        emoji = GENRE_EMOJI.get(top, "🎵")
        start = i * CLIP_DURATION
        badges += (
            f'<div class="clip-badge">'
            f'<span style="color:var(--muted);">Clip {i+1} · {start}s–{start+CLIP_DURATION}s</span>'
            f'&nbsp;→&nbsp;<span class="clip-badge-genre">{emoji} {top.capitalize()}</span>'
            f'&nbsp;<span style="color:var(--muted);">({cp[top]*100:.0f}%)</span>'
            f'</div>'
        )
    with st.expander(f"📊 Per-clip breakdown  ·  {len(per_clip)} clips analysed"):
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 0;">{badges}</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Core pipeline
# ─────────────────────────────────────────────
def run_full_pipeline(full_audio_path, label=""):
    _, _, gm = load_model_and_scaler()
    ng = len(gm)
    with st.spinner("Slicing audio into 30-second clips…"):
        clips = split_audio_into_clips(full_audio_path, CLIP_DURATION)
    if not clips:
        st.error("Could not split audio into clips."); return

    n   = len(clips)
    bar = st.progress(0)
    all_probs, per_clip = [], []
    for idx, path in enumerate(clips):
        bar.progress((idx+1)/n, text=f"Analysing clip {idx+1} / {n}…")
        probs = predict_single_clip(path)
        if probs is not None:
            all_probs.append(probs)
            per_clip.append({gm[i]: float(probs[i]) for i in range(ng)})
    bar.empty()
    for p in clips:
        try: os.unlink(p)
        except Exception: pass
    try: os.rmdir(os.path.dirname(clips[0]))
    except Exception: pass
    if not all_probs:
        st.error("All clips failed to process."); return
    avg      = np.mean(all_probs, axis=0)
    best     = gm.get(int(np.argmax(avg)), "Unknown")
    avg_dict = {gm[i]: float(avg[i]) for i in range(ng)}
    render_result(best, avg_dict, len(all_probs))
    render_per_clip(per_clip)

def run_single_clip_result(probs_array):
    _, _, gm = load_model_and_scaler()
    ng = len(gm)
    prob_dict = {gm[i]: float(probs_array[i]) for i in range(ng)}
    best = gm.get(int(np.argmax(probs_array)), "Unknown")
    render_result(best, prob_dict, 1)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    ytdlp_ok, ffmpeg_ok = check_ytdlp_ffmpeg()
    youtube_ok = ytdlp_ok and ffmpeg_ok

    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div class="hero-badge">🎵 &nbsp; AI-Powered · GTZAN CNN · 10 Genres</div>
    <h1 class="hero-title">SoundPrint</h1>
    <p class="hero-sub">
      Paste a Spotify link or drop a file — our CNN listens across the
      full song, clips by clip, and fingerprints its genre in seconds.
    </p>
    <div class="waveform-deco">
      <span></span><span></span><span></span><span></span><span></span>
      <span></span><span></span><span></span><span></span><span></span>
    </div>
    """, unsafe_allow_html=True)

    if not youtube_ok:
        st.markdown("""
        <div class="sp-card" style="border-color:rgba(245,158,11,0.25);background:rgba(245,158,11,0.04);">
          <p style="margin:0;font-size:0.84rem;color:#f59e0b;">
            ⚠️ <strong>Full-song analysis disabled.</strong>
            &nbsp;Install <code>yt-dlp</code> + <code>ffmpeg</code> for multi-clip ensemble prediction.
          </p>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────
    tab_spotify, tab_upload = st.tabs(["🎧  Spotify Link", "📂  Upload File"])

    # ── TAB 1: SPOTIFY ────────────────────────
    with tab_spotify:
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        spotify_url = st.text_input(
            "Spotify Track URL",
            placeholder="https://open.spotify.com/track/…",
        )
        st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
        go = st.button("✦  Classify Track", key="btn_spotify")

        if go and spotify_url:
            track_id = extract_track_id(spotify_url)
            if not track_id:
                st.error("Couldn't parse a track ID — please check the URL."); st.stop()

            with st.spinner("Connecting to Spotify…"):
                token = get_spotify_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
            if not token:
                st.error("Spotify authentication failed."); st.stop()

            with st.spinner("Fetching track metadata…"):
                info = get_track_info(track_id, token)
            if not info:
                st.error("Couldn't retrieve track info."); st.stop()

            track_name   = info.get("name", "Unknown")
            artists      = ", ".join(a["name"] for a in info.get("artists", []))
            album        = info.get("album", {}).get("name", "Unknown")
            album_imgs   = info.get("album", {}).get("images") or [{}]
            album_art    = album_imgs[0].get("url", "")
            preview_url  = info.get("preview_url")
            spotify_link = info.get("external_urls", {}).get("spotify", spotify_url)
            dur_ms       = info.get("duration_ms", 0)
            dur_str      = f"{dur_ms//60000}:{(dur_ms%60000)//1000:02d}" if dur_ms else "—"
            popularity   = info.get("popularity", "—")

            art_tag = (f'<img src="{album_art}" style="width:88px;height:88px;'
                       f'object-fit:cover;border-radius:12px;flex-shrink:0;'
                       f'box-shadow:0 8px 24px rgba(0,0,0,0.55);">'
                       if album_art else "")

            st.markdown(f"""
            <div class="track-card">
              {art_tag}
              <div class="track-meta" style="flex:1;min-width:0;">
                <h3>{track_name}</h3>
                <p>🎤 {artists}</p>
                <p>💿 {album}</p>
                <div class="stat-row" style="margin-top:8px;">
                  <div class="stat-pill">⏱ <strong>{dur_str}</strong></div>
                  <div class="stat-pill">🔥 <strong>{popularity}</strong> popularity</div>
                </div>
                <a href="{spotify_link}" target="_blank">↗ Open in Spotify</a>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if youtube_ok:
                st.info("🔄 Downloading full song from YouTube for multi-clip analysis…")
                with st.spinner(f"Fetching '{track_name}' from YouTube…"):
                    full_path = download_full_youtube(track_name, artists)

                if full_path and os.path.exists(full_path):
                    st.success("✅ Full song ready — running ensemble prediction.")
                    try:
                        run_full_pipeline(full_path, track_name)
                    finally:
                        try: os.unlink(full_path); os.rmdir(os.path.dirname(full_path))
                        except Exception: pass
                elif preview_url:
                    st.warning("YouTube failed — using Spotify 30s preview as fallback.")
                    st.audio(preview_url, format="audio/mp3")
                    data = download_preview(preview_url)
                    if data:
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
                            t.write(data); p = t.name
                        probs = predict_single_clip(p); os.unlink(p)
                        if probs is not None: run_single_clip_result(probs)
                else:
                    st.error("Both YouTube and Spotify preview failed. Try the Upload tab.")
            else:
                if preview_url:
                    st.info("✅ Using Spotify's 30-second preview (single clip).")
                    st.audio(preview_url, format="audio/mp3")
                    data = download_preview(preview_url)
                    if data:
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
                            t.write(data); p = t.name
                        probs = predict_single_clip(p); os.unlink(p)
                        if probs is not None: run_single_clip_result(probs)
                else:
                    st.markdown("""
                    <div class="sp-card" style="border-color:rgba(245,158,11,0.3);">
                      <p style="margin:0;color:#f59e0b;font-size:0.88rem;">
                        ⚠️ No Spotify preview &amp; yt-dlp not installed.<br>
                        <span style="color:var(--muted);font-size:0.8rem;">
                          Run <code>pip install yt-dlp</code> and install <code>ffmpeg</code>,
                          then restart — or use the Upload tab.
                        </span>
                      </p>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 2: UPLOAD ─────────────────────────
    with tab_upload:
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--muted);font-size:0.85rem;margin-bottom:0.8rem;">'
            'Upload a <strong style="color:var(--text);">.wav</strong> or '
            '<strong style="color:var(--text);">.mp3</strong> file. '
            'The full audio is split into 30-second clips, each clip is classified, '
            'and all predictions are averaged for the final result.</p>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drop your audio file here",
            type=["wav","mp3"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = ".wav" if uploaded.name.endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
                t.write(uploaded.read()); tmp_path = t.name
            st.audio(tmp_path, format=f"audio/{suffix.lstrip('.')}")
            if youtube_ok:
                try: run_full_pipeline(tmp_path, uploaded.name)
                finally:
                    try: os.unlink(tmp_path)
                    except Exception: pass
            else:
                probs = predict_single_clip(tmp_path)
                try: os.unlink(tmp_path)
                except Exception: pass
                if probs is not None: run_single_clip_result(probs)

    # ── Footer ────────────────────────────────
    st.markdown("""
    <div class="sp-footer">
      SoundPrint &nbsp;·&nbsp; CNN trained on GTZAN &nbsp;·&nbsp; 10 genres &nbsp;·&nbsp;
      Spotify API + yt-dlp ensemble &nbsp;·&nbsp; Made with <span>♥</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()