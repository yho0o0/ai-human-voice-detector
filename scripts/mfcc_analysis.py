from pathlib import Path

import librosa
import matplotlib
import numpy as np

# Use a non-interactive backend so plots can be saved in terminal/headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Define input folders and output folder using pathlib.
ai_dir = Path("data/ai")
human_dir = Path("data/human")
plots_dir = Path("plots")

# Create the plots folder if it does not exist.
plots_dir.mkdir(parents=True, exist_ok=True)


def compute_file_mfcc_vectors(audio_paths, start_time=2.0, end_time=7.0, n_mfcc=13):
    """
    For each audio file:
    1) load audio
    2) keep only the segment from start_time to end_time
    3) compute MFCCs
    4) average MFCCs over time
    Returns:
      - a 2D array with shape (num_files, n_mfcc)
      - sample rate from the first file (all files assumed to share sample rate)
    """
    mfcc_vectors = []
    sample_rate = None

    for audio_path in audio_paths:
        # Load one WAV file at its original sampling rate.
        signal, sr = librosa.load(audio_path, sr=None)

        # Keep sample rate from the first file for reference.
        if sample_rate is None:
            sample_rate = sr

        # Convert 2s and 7s into sample indices.
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Slice the signal to keep only the requested segment.
        segment = signal[start_sample:end_sample]

        # Compute 13 MFCC coefficients over time.
        # Result shape is (n_mfcc, num_frames).
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

        # Average each MFCC coefficient over time (frames).
        # This gives one vector of length n_mfcc for this file.
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_vectors.append(mfcc_mean)

    return np.array(mfcc_vectors), sample_rate


# Collect all WAV files from each group.
ai_paths = sorted(ai_dir.glob("*.wav"))
human_paths = sorted(human_dir.glob("*.wav"))

# Compute per-file MFCC vectors for both groups.
ai_mfcc_matrix, sr = compute_file_mfcc_vectors(ai_paths)
human_mfcc_matrix, _ = compute_file_mfcc_vectors(human_paths)

# Compute group-average MFCC vectors.
ai_group_mean = np.mean(ai_mfcc_matrix, axis=0)
human_group_mean = np.mean(human_mfcc_matrix, axis=0)

# x-axis positions for 13 coefficients.
mfcc_indices = np.arange(1, 14)


# Figure 1: line plot comparing group-average MFCC vectors (AI vs Human).
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(mfcc_indices, ai_group_mean, marker="o", label="AI Group Average")
ax1.plot(mfcc_indices, human_group_mean, marker="o", label="Human Group Average")
ax1.set_title("Average MFCC Comparison: AI vs Human")
ax1.set_xlabel("MFCC Coefficient Index")
ax1.set_ylabel("Average MFCC Value")
ax1.set_xticks(mfcc_indices)
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()
fig1.savefig(plots_dir / "mfcc_average_comparison.png", dpi=300)
plt.close(fig1)


# Figure 2: heatmap of all AI files (rows: files, columns: MFCC coefficients).
fig2, ax2 = plt.subplots(figsize=(10, 6))
im_ai = ax2.imshow(ai_mfcc_matrix, aspect="auto", origin="lower", cmap="viridis")
ax2.set_title("AI Files MFCC Heatmap")
ax2.set_xlabel("MFCC Coefficient Index")
ax2.set_ylabel("AI File Index")
ax2.set_xticks(np.arange(13))
ax2.set_xticklabels(mfcc_indices)
fig2.colorbar(im_ai, ax=ax2, label="MFCC Value")
fig2.tight_layout()
fig2.savefig(plots_dir / "mfcc_ai_heatmap.png", dpi=300)
plt.close(fig2)


# Figure 3: heatmap of all human files (rows: files, columns: MFCC coefficients).
fig3, ax3 = plt.subplots(figsize=(10, 6))
im_human = ax3.imshow(human_mfcc_matrix, aspect="auto", origin="lower", cmap="magma")
ax3.set_title("Human Files MFCC Heatmap")
ax3.set_xlabel("MFCC Coefficient Index")
ax3.set_ylabel("Human File Index")
ax3.set_xticks(np.arange(13))
ax3.set_xticklabels(mfcc_indices)
fig3.colorbar(im_human, ax=ax3, label="MFCC Value")
fig3.tight_layout()
fig3.savefig(plots_dir / "mfcc_human_heatmap.png", dpi=300)
plt.close(fig3)
