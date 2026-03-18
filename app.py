"""Streamlit app for AI-vs-Human voice detection using dynamic MFCC + enhanced features."""

import base64
import io
from pathlib import Path
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Training data folders.
AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")

# Feature settings.
N_MFCC = 13
SAMPLE_RATE = 16000
SEGMENT_START_SEC = 2.0
SEGMENT_END_SEC = 7.0
RANDOM_STATE = 42


def get_image_base64(image_path: Path) -> str | None:
    """Return a base64-encoded image string, or None if file does not exist."""
    if not image_path.exists():
        return None
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) as Python floats for any numeric array."""
    return float(np.mean(values)), float(np.std(values))


def extract_dynamic_feature_from_segment(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract the same dynamic feature vector used in scripts/dynamic_feature_classifier.py.
    Feature vector:
    - MFCC mean/std (13 + 13)
    - delta MFCC mean/std (13 + 13)
    - delta-delta MFCC mean/std (13 + 13)
    - centroid mean/std
    - bandwidth mean/std
    - flatness mean/std
    - zero-crossing-rate mean/std
    - RMS mean/std
    - pitch mean/std
    """
    # ----- A) Dynamic MFCC block -----
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)

    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    mfcc_dynamic_features = np.concatenate(
        [mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std]
    )

    # ----- B) Enhanced acoustic block -----
    spec_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    centroid_mean, centroid_std = safe_mean_std(spec_centroid)

    # 3) Spectral bandwidth.
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    bandwidth_mean, bandwidth_std = safe_mean_std(spec_bandwidth)

    # 4) Spectral flatness.
    spec_flatness = librosa.feature.spectral_flatness(y=segment)
    flatness_mean, flatness_std = safe_mean_std(spec_flatness)

    # 5) Zero-crossing rate.
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    zcr_mean, zcr_std = safe_mean_std(zcr)

    # 6) RMS energy.
    rms = librosa.feature.rms(y=segment)
    rms_mean, rms_std = safe_mean_std(rms)

    # 7) Pitch (YIN).
    pitch = librosa.yin(segment, fmin=50, fmax=500, sr=sr)
    pitch_mean, pitch_std = safe_mean_std(pitch)

    enhanced_features = np.array(
        [
            centroid_mean,
            centroid_std,
            bandwidth_mean,
            bandwidth_std,
            flatness_mean,
            flatness_std,
            zcr_mean,
            zcr_std,
            rms_mean,
            rms_std,
            pitch_mean,
            pitch_std,
        ],
        dtype=np.float64,
    )
    feature_vector = np.concatenate([mfcc_dynamic_features, enhanced_features])
    return feature_vector


def extract_dynamic_feature(wav_path: Path) -> np.ndarray:
    """Load one WAV file and extract dynamic feature vector."""
    segment, sr = extract_segment(wav_path)
    return extract_dynamic_feature_from_segment(segment, sr)


def extract_segment(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Load one WAV file with shared sample rate and keep only the 2~7 second segment.
    This matches the preprocessing assumptions in enhanced_classifier.py.
    """
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    start_idx = int(SEGMENT_START_SEC * sr)
    end_idx = int(SEGMENT_END_SEC * sr)
    segment = audio[start_idx:end_idx]
    if segment.size == 0:
        segment = audio
    return segment, sr


def figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_spectrogram_image(segment: np.ndarray, sr: int) -> str:
    """Create spectrogram image from the same 2~7 second segment."""
    stft_matrix = librosa.stft(segment)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    duration_sec = len(segment) / sr if sr > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    image = ax.imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration_sec, 0, sr / 2],
    )
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("dB")
    return figure_to_base64(fig)


def build_mfcc_heatmap_image(segment: np.ndarray, sr: int) -> str:
    """Create MFCC heatmap image from the same 2~7 second segment."""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    duration_sec = len(segment) / sr if sr > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    image = ax.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, duration_sec, 1, N_MFCC],
    )
    ax.set_title("MFCC Heatmap")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Index")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Coefficient")
    return figure_to_base64(fig)


def build_training_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full training dataset:
    - X: dynamic + enhanced feature vectors
    - y: labels (0=AI, 1=Human)
    """
    x_features: list[np.ndarray] = []
    y_labels: list[int] = []

    # Read all AI files and assign label 0.
    for wav_path in sorted(AI_DIR.glob("*.wav")):
        x_features.append(extract_dynamic_feature(wav_path))
        y_labels.append(0)

    # Read all Human files and assign label 1.
    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        x_features.append(extract_dynamic_feature(wav_path))
        y_labels.append(1)

    x = np.array(x_features, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return x, y


@st.cache_resource
def train_model() -> VotingClassifier:
    """
    Train an ensemble model once and cache it.
    Streamlit re-runs script often, so caching avoids retraining every click.
    """
    x_train, y_train = build_training_dataset()

    # Base model 1: Logistic Regression (with scaling).
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )

    # Base model 2: SVM with RBF kernel (with scaling).
    svm_rbf_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )

    # Base model 3: Random Forest.
    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Final model: Soft voting ensemble (average predicted probabilities).
    ensemble_model = VotingClassifier(
        estimators=[
            ("lr", logistic_model),
            ("svm", svm_rbf_model),
            ("rf", random_forest_model),
        ],
        voting="soft",
        n_jobs=-1,
    )

    ensemble_model.fit(x_train, y_train)
    return ensemble_model


def main() -> None:
    st.set_page_config(page_title="AI vs Human Voice Detector", layout="wide", menu_items={"Get Help": None, "Report a Bug": None, "About": None,})
    hide_streamlit_style = """<style>/* 우측 상단 Fork / GitHub / Deploy 영역 숨기기 */[data-testid="stToolbar"]{display: none;}/* 상단 기본 헤더 숨기기 */[data-testid="stHeader"]{display: none;}footer {visibility: hidden;}</style>"""st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # Custom CSS for a clean, portfolio-like interface.
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            }
            .block-container {
                max-width: 980px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }
            .hero-card, .result-card, .dev-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1.2rem 1.3rem;
                box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
            }
            .hero-title {
                margin: 0;
                color: #0f172a;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }
            .hero-subtext {
                margin-top: 0.5rem;
                margin-bottom: 0.1rem;
                color: #475569;
                line-height: 1.55;
            }
            .section-title {
                margin: 0 0 0.75rem 0;
                color: #0f172a;
                font-size: 1.15rem;
                font-weight: 650;
            }
            .result-line {
                margin: 0.35rem 0;
                color: #1e293b;
                font-size: 1rem;
                font-size: 1rem;
            }
            .prob-chip {
                display: inline-block;
                border-radius: 999px;
                padding: 0.22rem 0.7rem;
                margin-right: 0.4rem;
                margin-top: 0.35rem;
                font-size: 0.92rem;
                font-weight: 600;
            }
            .prob-chip-ai {
                background: #fee2e2;
                border: 1px solid #fecaca;
                color: #991b1b;
            }
            .prob-chip-human {
                background: #dcfce7;
                border: 1px solid #bbf7d0;
                color: #166534;
            }
            .prob-row {
                margin-top: 0.35rem;
                margin-bottom: 0.6rem;
            }
            .analysis-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.8rem;
                margin-top: 0.85rem;
            }
            .result-alert {
                margin-top: 0.3rem;
                margin-bottom: 0.1rem;
                padding: 0.55rem 0.7rem;
                border-radius: 10px;
                background: #fff1f2;
                border: 1px solid #fecdd3;
                color: #9f1239;
                font-size: 0.92rem;
                line-height: 1.45;
                font-weight: 600;
            }
            .result-alert-spacer {
                height: 0.45rem;
            }
            .analysis-item {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: #f8fafc;
                padding: 0.5rem;
            }
            .analysis-item img {
                width: 100%;
                border-radius: 8px;
                display: block;
            }
            .dev-wrap {
                display: grid;
                grid-template-columns: 140px 1fr;
                gap: 1rem;
                align-items: center;
            }
            .dev-card-title {
                margin: 0 0 0.8rem 0;
                color: #0f172a;
                font-size: 1.1rem !important;
                font-weight: 700;
                line-height: 1.1;
            }
            .note-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
            }
            .note-title {
                margin: 0 0 0.35rem 0;
                color: #0f172a;
                font-size: 1.1rem;
                font-weight: 700;
            }
            .note-text {
                margin: 0;
                color: #475569;
                line-height: 1.5;
                font-size: 0.92rem;
            }
            .dev-photo {
                width: 132px;
                height: 132px;
                object-fit: cover;
                border-radius: 14px;
                border: 1px solid #e5e7eb;
                background: #f1f5f9;
            }
            .dev-photo-missing {
                width: 132px;
                height: 132px;
                border-radius: 14px;
                border: 1px dashed #cbd5e1;
                background: #f8fafc;
                color: #64748b;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                font-size: 0.82rem;
                padding: 0.4rem;
            }
            .dev-name {
                margin: 0;
                color: #0f172a;
                font-size: 1.2rem;
                font-weight: 700;
            }
            .dev-affiliation {
                margin: 0.25rem 0 0.45rem 0;
                color: #334155;
                font-size: 0.95rem;
            }
            .dev-desc {
                margin: 0 0 0.8rem 0;
                color: #475569;
                line-height: 1.5;
                font-size: 0.92rem;
            }
            .contact-btn {
                display: inline-block;
                text-decoration: none !important;
                color: #0f172a !important;
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 999px;
                padding: 0.35rem 0.8rem;
                margin-right: 0.45rem;
                font-size: 0.9rem;
                font-weight: 600;
            }
            .contact-btn:hover {
                background: #eef2ff;
                border-color: #a5b4fc;
            }
            @media (max-width: 700px) {
                .analysis-grid {
                    grid-template-columns: 1fr;
                }
                .dev-wrap {
                    grid-template-columns: 1fr;
                    justify-items: start;
                }
                .dev-photo, .dev-photo-missing {
                    width: 112px;
                    height: 112px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">AI vs Human Voice Detector 🤖</h1>
            <p class="hero-subtext">
                AI-generated or human? Upload a WAV file. This system analyzes both static and time-varying acoustic patterns using MFCC, delta features, spectral, energy, and pitch-based descriptors to estimate the likelihood of AI-generated speech.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Train or load the cached classifier.
    model = train_model()

    # Uploader for one WAV file.
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:
        # Show uploaded audio in Streamlit audio player.
        st.audio(uploaded_file, format="audio/wav")

        # Save uploaded bytes to a temporary .wav file so librosa can load it.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        # Extract dynamic features from uploaded file with the same training pipeline.
        segment, sr = extract_segment(tmp_path)
        input_feature = extract_dynamic_feature_from_segment(segment, sr).reshape(1, -1)

        # Predict class and probabilities.
        predicted_class = int(model.predict(input_feature)[0])
        probabilities = model.predict_proba(input_feature)[0]
        spectrogram_image = build_spectrogram_image(segment, sr)
        mfcc_heatmap_image = build_mfcc_heatmap_image(segment, sr)

        # Convert class id to readable label.
        predicted_label = "AI" if predicted_class == 0 else "Human"
        ai_alert_html = (
            "<p class='result-alert'>⚠️ This voice shows characteristics similar to AI-generated speech. Please verify identity before taking action.</p>"
            if predicted_class == 0
            else "<div class='result-alert-spacer'></div>"
        )

        # Result card for prediction output.
        st.markdown(
            f"""
            <div class="result-card">
                <h3 class="section-title">Prediction Result</h3>
                <p class="result-line">Predicted label: <strong>{predicted_label}</strong></p>
                <div class="prob-row">
                    <span class="prob-chip prob-chip-ai">AI: {probabilities[0] * 100:.2f}%</span>
                    <span class="prob-chip prob-chip-human">Human: {probabilities[1] * 100:.2f}%</span>
                </div>
                {ai_alert_html}
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <img src="data:image/png;base64,{spectrogram_image}" alt="Spectrogram" />
                    </div>
                    <div class="analysis-item">
                        <img src="data:image/png;base64,{mfcc_heatmap_image}" alt="MFCC heatmap" />
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
        <div class="result-card">
            <h3 class="section-title" style="font-size:1.1rem;">How it works ⚙️</h3>
            <p class="result-line">1. Extract the 2-7 second speech segment</p>
            <p class="result-line">2. Compute dynamic acoustic features, including MFCC statistics, delta / delta-delta MFCC, spectral, energy, and pitch-based descriptors</p>
            <p class="result-line">3. Use a trained machine learning model to estimate whether the voice is more likely AI-generated or human</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="note-card">
            <p class="note-title">Note ❗️</p>
            <p class="note-text">
                Current model is trained on a limited dataset and may be sensitive to speaker diversity and recording conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Developer profile card (portfolio-style section).
    profile_path = Path("assets/profile.jpg")
    profile_base64 = get_image_base64(profile_path)
    if profile_base64:
        profile_html = (
            f'<img class="dev-photo" src="data:image/jpeg;base64,{profile_base64}" '
            'alt="YeongHo Lee profile photo" />'
        )
    else:
        # Show a small notice instead of crashing if image is missing.
        profile_html = '<div class="dev-photo-missing">Profile image not found<br/>assets/profile.jpg</div>'

    st.markdown(
        f"""
        <div class="dev-card">
            <p class="dev-card-title" style="font-size:1.1rem; line-height:1.1;">Developer 💻</p>
            <div class="dev-wrap">
                <div>{profile_html}</div>
                <div>
                    <p class="dev-name">YeongHo Lee</p>
                    <p class="dev-affiliation">Department of Mechanical Engineering, Chung-Ang University</p>
                    <a class="contact-btn" href="mailto:qsx0101@icloud.com">Email</a>
                    <a class="contact-btn" href="https://www.linkedin.com/in/yeongho-lee-203162206/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
