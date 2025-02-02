"""
Live FFT from Microphone using Matplotlib

Press the 'd' key in the plot window to toggle between two display modes:
    - Mode 0: Top three frequency peaks are highlighted.
    - Mode 1: Only the single maximum frequency is highlighted.
"""

import pyaudio
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Audio settings
CHUNK = 1024          # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit int format
CHANNELS = 1          # Mono audio
RATE = 44100          # Sampling rate (Hz)

# FFT and display settings
FFT_SIZE = CHUNK      # We'll perform FFT on each chunk
FILTER_THRESHOLD = 2  # Values below this are set to zero
NUM_MAXES = 3         # Number of peaks to highlight in mode 0

# Global variable for display mode
# display_mode = 0: highlight top three peaks
# display_mode = 1: highlight only the single maximum peak
display_mode = 0

def on_key_press(event):
    """Toggle display mode when 'd' key is pressed."""
    global display_mode
    if event.key == 'd':
        display_mode = 1 - display_mode
        print(f"Display mode changed to {display_mode}.")

def main():
    global display_mode

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Create a Hanning window to reduce spectral leakage
    window = np.hanning(FFT_SIZE)

    # Set up Matplotlib figure and bar container.
    # We only display the positive frequency components (first half of FFT).
    freqs = np.linspace(0, RATE/2, FFT_SIZE//2)
    fig, ax = plt.subplots()
    bars = ax.bar(freqs, np.zeros(FFT_SIZE//2), width=freqs[1]-freqs[0], color='r')
    ax.set_xlim(0, RATE/2)
    ax.set_ylim(0, 100)  # Adjust as needed based on observed FFT amplitude
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live FFT from Microphone")

    # Connect the key press event to toggle display modes
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    def update_frame(frame):
        # Read a chunk of data from the microphone.
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print("Error reading audio stream:", e)
            return

        # Convert binary data to a NumPy array of signed 16-bit integers.
        audio_data = np.array(struct.unpack(f"{CHUNK}h", data))
        # Apply windowing
        windowed_data = audio_data * window

        # Compute the FFT and only take the positive frequency components
        yf = np.fft.fft(windowed_data)
        N = len(yf)
        yff = (1.0 / N * np.abs(yf[:N//2]))

        # Optional filtering: zero out small values
        yff = np.array([val if val > FILTER_THRESHOLD else 0 for val in yff])

        # Prepare colors: default all bars are red.
        colors = ['r'] * (FFT_SIZE//2)

        if display_mode == 1:
            # Single maximum mode: highlight only the highest peak (ignoring DC)
            if len(yff) > 1:
                max_index = yff[1:].argmax() + 1  # offset by one for skipping DC component
                colors[max_index] = 'blue'
        elif display_mode == 0:
            # Top three mode: highlight the three largest peaks (ignoring DC)
            # Make a copy so that we can remove the max without altering original yff.
            find_max = yff[1:].copy()
            # Colors for the top three (you can customize these)
            highlight_colors = ['blue', 'green', 'orange']
            for i in range(min(NUM_MAXES, len(find_max))):
                idx = find_max.argmax() + 1
                colors[idx] = highlight_colors[i]
                find_max[idx-1] = -1  # set to -1 so it's not chosen again

        # Update bar heights and colors.
        for bar, height, col in zip(bars, yff, colors):
            bar.set_height(height)
            bar.set_color(col)

        return bars

    # Use FuncAnimation to update the plot in real time.
    ani = animation.FuncAnimation(fig, update_frame, interval=10, blit=False)

    plt.show()

    # Clean up after the plot window is closed.
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
