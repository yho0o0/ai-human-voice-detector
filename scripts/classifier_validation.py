"""Validate an AI-vs-Human audio classifier with MFCC features and 5-fold CV."""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Constants to keep the script easy to adjust.
AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")
N_MFCC = 13
SEGMENT_START_SEC = 2.0
SEGMENT_END_SEC = 7.0
N_SPLITS = 5
RANDOM_STATE = 42


def extract_mfcc_feature(wav_path: Path, sample_rate: int | None = None) -> np.ndarray:
    """
    Load one WAV file, keep only 2s~7s, compute 13 MFCCs,
    then average each MFCC over time to get a 13D vector.
    """
    # Load the waveform. If sample_rate is None, librosa keeps the original rate.
    audio, sr = librosa.load(wav_path, sr=sample_rate)

    # Convert seconds to sample indices and slice the selected segment.
    start_idx = int(SEGMENT_START_SEC * sr)
    end_idx = int(SEGMENT_END_SEC * sr)
    audio_segment = audio[start_idx:end_idx]

    # MFCC shape is (n_mfcc, number_of_frames).
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=N_MFCC)

    # Average over time axis so each file becomes one 13-dimensional vector.
    feature_vector = np.mean(mfcc, axis=1)
    return feature_vector


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build dataset arrays:
    - X: MFCC feature vectors
    - y: labels (0=AI, 1=Human)
    """
    x_features: list[np.ndarray] = []
    y_labels: list[int] = []

    # Load AI files and assign label 0.
    for wav_path in sorted(AI_DIR.glob("*.wav")):
        x_features.append(extract_mfcc_feature(wav_path))
        y_labels.append(0)

    # Load Human files and assign label 1.
    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        x_features.append(extract_mfcc_feature(wav_path))
        y_labels.append(1)

    x = np.array(x_features, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return x, y


def plot_feature_importance(coefficients: np.ndarray) -> None:
    """Plot absolute Logistic Regression coefficients for MFCC1~MFCC13."""
    # Use absolute values to show importance magnitude regardless of sign.
    importance = np.abs(coefficients)
    feature_names = [f"MFCC {i}" for i in range(1, N_MFCC + 1)]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(feature_names, importance, color="steelblue")
    plt.title("Feature Importance from Logistic Regression Coefficients")
    plt.xlabel("MFCC Coefficient")
    plt.ylabel("Absolute Coefficient Value")
    plt.xticks(rotation=45)

    # Add value labels on top of each bar for readability.
    for bar, value in zip(bars, importance):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Step 1~5: Build X and y from wav files.
    x, y = load_dataset()

    # Build a pipeline so scaling happens inside each CV fold correctly.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )

    # Step 7: 5-fold cross validation.
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = cross_val_score(model, x, y, cv=kfold, scoring="accuracy")

    # Step 8: Print fold accuracies, mean, and standard deviation.
    print("5-Fold Cross Validation Accuracy:")
    for fold_idx, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {fold_idx}: {acc:.4f}")

    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")

    # Step 9: Fit on the full dataset.
    model.fit(x, y)

    # Step 10: Extract model coefficients and plot feature importance.
    coefficients = model.named_steps["classifier"].coef_[0]
    print("\nLogistic Regression Coefficients:")
    for i, coef in enumerate(coefficients, start=1):
        print(f"MFCC {i}: {coef:.6f}")

    plot_feature_importance(coefficients)


if __name__ == "__main__":
    main()
