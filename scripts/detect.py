"""Train a simple AI-vs-Human classifier and predict one input WAV file."""

from pathlib import Path
import sys

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Paths to the training folders.
AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")

# Audio and feature settings.
N_MFCC = 13
SEGMENT_START_SEC = 2.0
SEGMENT_END_SEC = 7.0
RANDOM_STATE = 42


def extract_mfcc_feature(wav_path: Path, sample_rate: int | None = None) -> np.ndarray:
    """
    Load one WAV file, keep only the 2s~7s segment, compute 13 MFCCs,
    and average over time to get one 13-dimensional feature vector.
    """
    # Load audio. If sample_rate is None, librosa keeps the file's original rate.
    audio, sr = librosa.load(wav_path, sr=sample_rate)

    # Convert the target segment (2s~7s) into sample indices.
    start_idx = int(SEGMENT_START_SEC * sr)
    end_idx = int(SEGMENT_END_SEC * sr)
    segment = audio[start_idx:end_idx]

    # Compute MFCCs for the segment. Output shape: (13, num_frames).
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)

    # Average each MFCC row across time to get a fixed-size 13D vector.
    feature_vector = np.mean(mfcc, axis=1)
    return feature_vector


def build_training_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build training data:
    - X: MFCC feature vectors
    - y: labels (0=AI, 1=Human)
    """
    x_features: list[np.ndarray] = []
    y_labels: list[int] = []

    # Add AI samples with label 0.
    for wav_path in sorted(AI_DIR.glob("*.wav")):
        x_features.append(extract_mfcc_feature(wav_path))
        y_labels.append(0)

    # Add Human samples with label 1.
    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        x_features.append(extract_mfcc_feature(wav_path))
        y_labels.append(1)

    x = np.array(x_features, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int64)
    return x, y


def train_classifier(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train logistic regression on the full training dataset.
    A scaler is included so each MFCC feature is on a similar scale.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x, y)
    return model


def main() -> None:
    # Step 7: Read one input WAV path from the command line.
    if len(sys.argv) != 2:
        print("Usage: python scripts/detect.py <input_wav_path>")
        sys.exit(1)

    input_wav_path = Path(sys.argv[1])
    if not input_wav_path.exists():
        print(f"Error: file not found -> {input_wav_path}")
        sys.exit(1)

    # Step 1~6: Build training data and train logistic regression on full dataset.
    x_train, y_train = build_training_dataset()
    model = train_classifier(x_train, y_train)

    # Step 8: Extract features from the input WAV using the same process.
    input_feature = extract_mfcc_feature(input_wav_path)
    input_feature = input_feature.reshape(1, -1)  # Model expects shape: (n_samples, n_features)

    # Predict class label and class probabilities.
    predicted_class = int(model.predict(input_feature)[0])
    probabilities = model.predict_proba(input_feature)[0]

    # Convert numeric label to text label.
    predicted_label = "AI" if predicted_class == 0 else "Human"

    # Step 9: Print prediction and confidence for each class.
    print(f"Predicted label: {predicted_label}")
    print(f"Probability AI (class 0): {probabilities[0]:.4f}")
    print(f"Probability Human (class 1): {probabilities[1]:.4f}")


if __name__ == "__main__":
    main()
