from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np

# Use a non-interactive backend so plotting works in terminal/headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Define the folders that contain AI and human recordings.
ai_dir = Path("data/ai")
human_dir = Path("data/human")

# Define where to save the final figure.
output_plot_path = Path("plots/spectrogram_group_average.png")

# Create the output folder if it does not already exist.
output_plot_path.parent.mkdir(parents=True, exist_ok=True)


def compute_group_average_spectrogram(audio_paths, start_time, end_time):
    """
    Load all files in a group, convert each one into a dB spectrogram
    for the selected time window, then return the average spectrogram.
    """
    spectrograms_db = []
    group_sr = None

    for audio_path in audio_paths:
        # Load one audio file while keeping its original sample rate.
        signal, sr = librosa.load(audio_path, sr=None)

        # Store sample rate from the first file (assumed same for all files).
        if group_sr is None:
            group_sr = sr

        # Convert start/end times (seconds) to sample indices.
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Slice the signal to keep only the 2s to 7s segment.
        segment = signal[start_sample:end_sample]

        # Compute STFT and convert magnitude to decibels.
        stft = librosa.stft(segment)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        spectrograms_db.append(spectrogram_db)

    # Stack all spectrograms into one 3D array and average across files.
    # Axis 0 is "which file", so mean over axis 0 gives group average.
    average_spectrogram_db = np.mean(np.stack(spectrograms_db, axis=0), axis=0)
    return average_spectrogram_db, group_sr


# Get sorted lists of WAV files from both groups.
ai_audio_paths = sorted(ai_dir.glob("*.wav"))
human_audio_paths = sorted(human_dir.glob("*.wav"))

# Time window to analyze for every file.
start_time = 2.0
end_time = 7.0

# Compute average spectrograms for AI and human groups.
ai_avg_db, ai_sr = compute_group_average_spectrogram(
    ai_audio_paths, start_time, end_time
)
human_avg_db, human_sr = compute_group_average_spectrogram(
    human_audio_paths, start_time, end_time
)

# Create a figure with two subplots for side-by-side comparison by group.
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot average AI spectrogram.
ai_img = librosa.display.specshow(
    ai_avg_db,
    sr=ai_sr,
    x_axis="time",
    y_axis="hz",
    ax=axes[0],
)
axes[0].set_title("Average AI Spectrogram (2s to 7s)")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Frequency (Hz)")
fig.colorbar(ai_img, ax=axes[0], format="%+2.0f dB", label="Amplitude (dB)")

# Plot average human spectrogram.
human_img = librosa.display.specshow(
    human_avg_db,
    sr=human_sr,
    x_axis="time",
    y_axis="hz",
    ax=axes[1],
)
axes[1].set_title("Average Human Spectrogram (2s to 7s)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Frequency (Hz)")
fig.colorbar(human_img, ax=axes[1], format="%+2.0f dB", label="Amplitude (dB)")

# Adjust layout so subplot labels and colorbars do not overlap.
plt.tight_layout()

# Save the final figure.
plt.savefig(output_plot_path, dpi=300)
plt.close(fig)
