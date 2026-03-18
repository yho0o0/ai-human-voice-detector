from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def extract_mfcc_feature(file_path: Path, start_sec: float = 2.0, end_sec: float = 7.0) -> np.ndarray:
    """
    Load one WAV file, keep only the segment from start_sec to end_sec,
    compute 13 MFCCs, and return the time-averaged 13D feature vector.
    """
    # Load audio file. sr=None keeps the original sample rate of the file.
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Convert time (seconds) to sample indices.
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Safely clip indices so we don't go out of bounds for short files.
    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))

    # Extract only the requested audio segment.
    segment = audio[start_sample:end_sample]

    # If the segment is empty (very short file), return zeros as fallback.
    if len(segment) == 0:
        return np.zeros(13, dtype=np.float32)

    # Compute MFCC features. Output shape: (13, number_of_time_frames).
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)

    # Average over time frames to get one 13-dimensional vector.
    feature_vector = np.mean(mfcc, axis=1)
    return feature_vector


def main() -> None:
    # Build paths relative to the project root.
    project_root = Path(__file__).resolve().parent.parent
    ai_dir = project_root / "data" / "ai"
    human_dir = project_root / "data" / "human"

    # Collect all .wav files from each folder.
    ai_files = sorted(ai_dir.glob("*.wav"))
    human_files = sorted(human_dir.glob("*.wav"))

    # X will store feature vectors, y will store labels.
    X = []
    y = []

    # Process AI files and assign label 0.
    for wav_file in ai_files:
        X.append(extract_mfcc_feature(wav_file))
        y.append(0)

    # Process Human files and assign label 1.
    for wav_file in human_files:
        X.append(extract_mfcc_feature(wav_file))
        y.append(1)

    # Convert to NumPy arrays for scikit-learn.
    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets with an 80/20 ratio.
    # stratify=y keeps the class balance similar in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train a Logistic Regression classifier.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict labels for the test set.
    y_pred = model.predict(X_test)

    # Evaluate model performance.
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print requested metrics.
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
