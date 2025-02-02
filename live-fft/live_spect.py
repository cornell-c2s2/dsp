import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------

WAV_FILE = "audio/1809v2.wav"

# 1-second window, but hop forward by 0.25 seconds each time
CHUNK_DURATION = 1.0   # seconds (display window)
HOP_DURATION   = 0.05  # seconds (advance step)

# Spectrogram parameters
N_PER_SEG = 256        # FFT window size
OVERLAP   = 128        # Overlap in samples for the FFT

# -----------------------------------------------------------------
# Read WAV File
# -----------------------------------------------------------------
sr, data = wavfile.read(WAV_FILE)

# If stereo, convert to mono (optional)
if data.ndim > 1:
    data = data.mean(axis=1)

num_samples = len(data)

# Convert durations to samples
chunk_size = int(CHUNK_DURATION * sr)
hop_size   = int(HOP_DURATION   * sr)

# Number of animation frames
# Once we reach beyond the last chunk, stop
num_frames = (num_samples - chunk_size) // hop_size + 1
if num_frames < 1:
    num_frames = 1  # If the file is very short, ensure at least 1 frame

# -----------------------------------------------------------------
# Set up Matplotlib Figure
# -----------------------------------------------------------------
fig, ax = plt.subplots()
fig.suptitle("Sliding Spectrogram (1s window, 0.05s hop)", fontsize=14)

# Initialize an empty image; we'll update it each frame
im = ax.imshow(
    np.zeros((N_PER_SEG//2+1, 1)),  # dummy shape to start
    origin='lower',
    aspect='auto',
    interpolation='nearest',
    cmap='viridis'
)
ax.set_xlabel("Time (in FFT frames)")
ax.set_ylabel("Frequency Bin")

# -----------------------------------------------------------------
# Update Function for Animation
# -----------------------------------------------------------------
def update_spectrogram(frame_idx):
    """
    Called by FuncAnimation for each new frame.
    Displays 1 second of data, starting at frame_idx * hop_size.
    """
    start = frame_idx * hop_size
    end   = start + chunk_size

    # Safety check (clamp to length)
    if end > num_samples:
        end = num_samples

    # Extract the chunk of audio
    chunk_data = data[start:end]

    # Compute spectrogram
    f, t, Sxx = spectrogram(
        chunk_data,
        fs=sr,
        nperseg=N_PER_SEG,
        noverlap=OVERLAP
    )

    # Convert to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Update image data
    im.set_data(Sxx_dB)

    # Update axes extents:
    # - x-dimension goes from 0 to Sxx_dB.shape[1] frames (or you can label in actual time).
    # - y-dimension goes from 0 to the number of frequency bins.
    im.set_extent((0, Sxx_dB.shape[1], 0, Sxx_dB.shape[0]))

    # Update color scale
    im.set_clim(vmin=Sxx_dB.min(), vmax=Sxx_dB.max())

    # Optionally, update axis labels or title
    ax.set_title(f"Time Window: {start/sr:.2f}s to {end/sr:.2f}s")

    return [im]

# -----------------------------------------------------------------
# Create the Animation
# -----------------------------------------------------------------
ani = FuncAnimation(
    fig,
    update_spectrogram,
    frames=num_frames,
    interval=int(HOP_DURATION * 1000),  # update ~ every 250ms
    blit=False
)

# -----------------------------------------------------------------
# Display the Plot
# -----------------------------------------------------------------
plt.show()
