"""Build a stronger AI-vs-Human classifier with dynamic MFCC + enhanced features."""

from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Project paths (this script lives in scripts/, so parents[1] is project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_ROOT / "data" / "ai"
HUMAN_DIR = PROJECT_ROOT / "data" / "human"

# Shared preprocessing settings.
SAMPLE_RATE = 16000
SEGMENT_START_SEC = 2.0
SEGMENT_END_SEC = 7.0
N_MFCC = 13
RANDOM_STATE = 42
N_SPLITS = 5


def extract_segment(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Load one WAV file as mono, resample to shared sample rate,
    and keep only the 2~7 second segment.
    """
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    start_idx = int(SEGMENT_START_SEC * sr)
    end_idx = int(SEGMENT_END_SEC * sr)
    segment = audio[start_idx:end_idx]
    return segment, sr


def safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return mean and standard deviation as plain Python floats."""
    return float(np.mean(values)), float(np.std(values))


def extract_dynamic_feature_vector(wav_path: Path) -> np.ndarray:
    """
    Extract one feature vector with:
    A) Dynamic MFCC features:
       - MFCC mean/std (13 + 13)
       - delta MFCC mean/std (13 + 13)
       - delta-delta MFCC mean/std (13 + 13)
    B) Enhanced acoustic features:
       - centroid mean/std
       - bandwidth mean/std
       - flatness mean/std
       - zero-crossing-rate mean/std
       - RMS mean/std
       - pitch mean/std
    """
    segment, sr = extract_segment(wav_path)

    # ----- A) MFCC dynamic block -----
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Aggregate each coefficient across time using mean and std.
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

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    bandwidth_mean, bandwidth_std = safe_mean_std(spec_bandwidth)

    spec_flatness = librosa.feature.spectral_flatness(y=segment)
    flatness_mean, flatness_std = safe_mean_std(spec_flatness)

    zcr = librosa.feature.zero_crossing_rate(y=segment)
    zcr_mean, zcr_std = safe_mean_std(zcr)

    rms = librosa.feature.rms(y=segment)
    rms_mean, rms_std = safe_mean_std(rms)

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

    # Final vector = dynamic MFCC block + enhanced acoustic block.
    feature_vector = np.concatenate([mfcc_dynamic_features, enhanced_features])
    return feature_vector


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build full dataset:
    - X: feature vectors
    - y: labels (0=AI, 1=Human)
    """
    x_features: list[np.ndarray] = []
    y_labels: list[int] = []

    for wav_path in sorted(AI_DIR.glob("*.wav")):
        x_features.append(extract_dynamic_feature_vector(wav_path))
        y_labels.append(0)

    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        x_features.append(extract_dynamic_feature_vector(wav_path))
        y_labels.append(1)

    x = np.array(x_features, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return x, y


def evaluate_model(model_name: str, model: Pipeline, x: np.ndarray, y: np.ndarray) -> None:
    """Evaluate one model with 5-fold stratified cross validation."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    print(f"\n=== {model_name} ===")
    print("Fold accuracies:")
    for fold_idx, score in enumerate(scores, start=1):
        print(f"  Fold {fold_idx}: {score:.4f}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Std accuracy : {scores.std():.4f}")


def main() -> None:
    """Run dataset build + model evaluation."""
    print("Building dataset from data/ai and data/human ...")
    x, y = build_dataset()

    # Model 1: Logistic Regression.
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )

    # Model 2: SVM with RBF kernel.
    svm_rbf_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )

    evaluate_model("Logistic Regression", logistic_model, x, y)
    evaluate_model("SVM (RBF)", svm_rbf_model, x, y)

    print(f"\nTotal number of features used: {x.shape[1]}")


if __name__ == "__main__":
    main()
