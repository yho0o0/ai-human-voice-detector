"""Train and evaluate richer AI-vs-Human voice classifiers with cross validation."""

from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Project paths (this script is inside scripts/, so parent[1] is project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_ROOT / "data" / "ai"
HUMAN_DIR = PROJECT_ROOT / "data" / "human"

# Audio and feature settings.
SAMPLE_RATE = 16000
SEGMENT_START_SEC = 2.0
SEGMENT_END_SEC = 7.0
N_MFCC = 13
RANDOM_STATE = 42
N_SPLITS = 5


def extract_segment(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio with a shared sample rate and return only the 2~7 second segment.
    We assume files are at least 8 seconds long as requested.
    """
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    start_idx = int(SEGMENT_START_SEC * sr)
    end_idx = int(SEGMENT_END_SEC * sr)
    segment = audio[start_idx:end_idx]
    return segment, sr


def safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) as Python floats for any 1D/2D numeric array."""
    return float(np.mean(values)), float(np.std(values))


def extract_richer_features(wav_path: Path) -> np.ndarray:
    """
    Extract richer features from one WAV file:
    - 13 MFCC means
    - mean/std of spectral centroid
    - mean/std of spectral bandwidth
    - mean/std of spectral flatness
    - mean/std of zero-crossing rate
    - mean/std of RMS energy
    - mean/std of pitch (YIN)
    """
    segment, sr = extract_segment(wav_path)

    # 1) MFCC (13 coefficients averaged over time).
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)  # Shape: (13,)

    # 2) Spectral centroid.
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

    # 7) Pitch (fundamental frequency) from YIN.
    # We limit pitch search range to typical speech range.
    pitch = librosa.yin(segment, fmin=50, fmax=500, sr=sr)
    pitch_mean, pitch_std = safe_mean_std(pitch)

    # Combine all features into one vector.
    feature_vector = np.concatenate(
        [
            mfcc_mean,
            np.array(
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
            ),
        ]
    )
    return feature_vector


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full dataset:
    - X: feature vectors
    - y: labels (0=AI, 1=Human)
    """
    x_features: list[np.ndarray] = []
    y_labels: list[int] = []

    for wav_path in sorted(AI_DIR.glob("*.wav")):
        x_features.append(extract_richer_features(wav_path))
        y_labels.append(0)

    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        x_features.append(extract_richer_features(wav_path))
        y_labels.append(1)

    x = np.array(x_features, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return x, y


def evaluate_model(model_name: str, model: Pipeline, x: np.ndarray, y: np.ndarray) -> None:
    """Run 5-fold cross validation and print fold scores + summary."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    print(f"\n=== {model_name} ===")
    print("Fold accuracies:")
    for fold_idx, score in enumerate(scores, start=1):
        print(f"  Fold {fold_idx}: {score:.4f}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Std accuracy : {scores.std():.4f}")


def main() -> None:
    """Build dataset, evaluate models with CV, then fit both on full data."""
    print("Building dataset from data/ai and data/human ...")
    x, y = build_dataset()

    # Two models requested: Logistic Regression and SVM (RBF).
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )
    svm_rbf_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )

    # Evaluate with 5-fold CV.
    evaluate_model("Logistic Regression", logistic_model, x, y)
    evaluate_model("SVM (RBF)", svm_rbf_model, x, y)

    # Fit both models on the full dataset after evaluation.
    logistic_model.fit(x, y)
    svm_rbf_model.fit(x, y)

    print("\nFitted both models on full dataset.")
    print(f"Total number of features used: {x.shape[1]}")


if __name__ == "__main__":
    main()
