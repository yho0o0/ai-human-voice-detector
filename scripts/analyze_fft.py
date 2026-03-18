from pathlib import Path

import librosa
import matplotlib
import numpy as np

# Use a non-GUI backend so saving the figure works in terminal/headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Input folders (each folder contains multiple .wav files).
AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")

# Output image path.
OUTPUT_PATH = Path("plots/fft_group_average.png")

# We only analyze this time range from each audio file.
START_SEC = 2.0
END_SEC = 7.0

# Use one shared sample rate so all FFT vectors have the same length.
TARGET_SR = 16_000


def get_wav_files(folder: Path) -> list[Path]:
    """Return all .wav files in a folder."""
    return sorted(folder.glob("*.wav"))


def extract_time_segment(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cut the audio to only the 2s to 7s segment."""
    start_idx = int(START_SEC * sample_rate)
    end_idx = int(END_SEC * sample_rate)
    return audio[start_idx:end_idx]


def compute_fft_magnitude(audio_segment: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT magnitude and its frequency axis."""
    # rfft keeps only non-negative frequencies (good for real-valued audio signals).
    fft_values = np.fft.rfft(audio_segment)

    # Magnitude tells us the strength of each frequency component.
    magnitude = np.abs(fft_values)

    # Build matching frequency values in Hz.
    frequencies = np.fft.rfftfreq(len(audio_segment), d=1.0 / sample_rate)
    return frequencies, magnitude


def average_group_fft(file_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    For one group of files:
    1) load each file
    2) extract 2s~7s
    3) compute FFT magnitude
    4) average magnitudes across files
    """
    all_magnitudes: list[np.ndarray] = []
    frequencies: np.ndarray | None = None

    for file_path in file_paths:
        # Load as mono and resample to a shared sample rate.
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # Keep only the requested time window.
        segment = extract_time_segment(audio, TARGET_SR)

        # FFT for this file's segment.
        freq, mag = compute_fft_magnitude(segment, TARGET_SR)
        frequencies = freq
        all_magnitudes.append(mag)

    # Average magnitudes at each frequency bin.
    avg_magnitude = np.mean(np.vstack(all_magnitudes), axis=0)
    return frequencies, avg_magnitude


def main() -> None:
    # Find all wav files in each group folder.
    ai_files = get_wav_files(AI_DIR)
    human_files = get_wav_files(HUMAN_DIR)

    if not ai_files:
        raise ValueError(f"No .wav files found in: {AI_DIR}")
    if not human_files:
        raise ValueError(f"No .wav files found in: {HUMAN_DIR}")

    # Compute group-average FFT magnitudes.
    ai_freqs, ai_avg_mag = average_group_fft(ai_files)
    human_freqs, human_avg_mag = average_group_fft(human_files)

    # Create 3 plots:
    # 1) AI average
    # 2) Human average
    # 3) Both averages on one graph for comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(ai_freqs, ai_avg_mag, color="tab:blue")
    axes[0].set_title("Average FFT Magnitude Spectrum (AI Group)")

    axes[1].plot(human_freqs, human_avg_mag, color="tab:orange")
    axes[1].set_title("Average FFT Magnitude Spectrum (Human Group)")

    axes[2].plot(ai_freqs, ai_avg_mag, color="tab:blue", label="AI Average")
    axes[2].plot(human_freqs, human_avg_mag, color="tab:orange", label="Human Average")
    axes[2].set_title("AI vs Human: Average FFT Magnitude Spectrum")
    axes[2].legend()

    # Formatting for readability and the requested x-axis limit.
    for ax in axes:
        ax.set_xlim(0, 8000)
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Frequency (Hz)")

    # Make sure the output folder exists, then save the figure.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
