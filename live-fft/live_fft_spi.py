"""
Live FFT from SPI using Matplotlib

Press the 'd' key in the plot window to toggle between two display modes:
    - Mode 0: Top three frequency peaks are highlighted.
    - Mode 1: Only the single maximum frequency is highlighted.

IMPORTANT:
- This is a template to demonstrate how one might switch from PyAudio to SPI.
- You must adapt SPI settings (bus, device, speed, data format, timing) to your hardware.
"""

import spidev
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# SPI and "audio" settings
CHUNK = 1024            # Number of samples to read per frame (adapt to your hardware)
SPI_BUS = 0             # SPI bus (change if needed)
SPI_DEVICE = 0          # SPI device/chip select (change if needed)
SPI_SPEED_HZ = 2_000_000  # SPI clock speed in Hz (adapt to your hardware)

# For audio-like FFT, set an expected sampling rate
# This should match your actual ADC sampling rate over SPI.
RATE = 44100  # in Hz (example: if your ADC is sampling at 44.1 kHz)
              # Adjust to whatever your real sampling rate is.

# FFT and display settings
FFT_SIZE = CHUNK
FILTER_THRESHOLD = 2
NUM_MAXES = 3

# Display mode: 0 or 1
display_mode = 0

def on_key_press(event):
    """Toggle display mode when 'd' key is pressed."""
    global display_mode
    if event.key == 'd':
        display_mode = 1 - display_mode
        print(f"Display mode changed to {display_mode}.")

def main():
    global display_mode

    # -----------------------------------------------------------------------
    # 1. Initialize SPI
    # -----------------------------------------------------------------------
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    spi.max_speed_hz = SPI_SPEED_HZ
    # Depending on your hardware, you may also need to configure
    # spi.mode (0, 1, 2, or 3), spi.bits_per_word, etc.

    # -----------------------------------------------------------------------
    # 2. Prepare the Hanning window
    # -----------------------------------------------------------------------
    window = np.hanning(FFT_SIZE)

    # -----------------------------------------------------------------------
    # 3. Set up Matplotlib figure
    # -----------------------------------------------------------------------
    freqs = np.linspace(0, RATE/2, FFT_SIZE//2)
    fig, ax = plt.subplots()
    bars = ax.bar(freqs, np.zeros(FFT_SIZE//2), width=freqs[1]-freqs[0], color='r')
    ax.set_xlim(0, RATE/2)
    ax.set_ylim(0, 100)  # Adjust if your amplitude range differs
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live FFT from SPI Driver")

    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -----------------------------------------------------------------------
    # 4. Function to read samples from SPI
    # -----------------------------------------------------------------------
    def read_spi_samples(num_samples):
        """
        Reads num_samples of 16-bit data from SPI.
        Returns a NumPy array of signed 16-bit integers.
        
        NOTE: This is a simplistic example: we send num_samples*2 dummy bytes
        (or 0x00) and expect to get num_samples*2 bytes back from the device.
        Many ADCs require specific command frames, register addresses, etc.
        Modify accordingly for your hardware protocol.
        """
        # Each 16-bit sample is 2 bytes
        bytes_to_read = num_samples * 2
        
        # Perform SPI transfer. We send all 0x00 to clock out data.
        # The device must be continuously sampling and ready to return data.
        raw_data = spi.xfer2([0x00] * bytes_to_read)

        # Convert the returned bytes to a NumPy array of signed 16-bit integers.
        # 'raw_data' is a list of ints (0-255).
        # We'll pack them into a binary string, then unpack as little-endian 16-bit.
        # Adjust endianness if your hardware uses big-endian format.
        byte_str = bytes(raw_data)
        samples = struct.unpack('<' + 'h' * num_samples, byte_str)
        
        return np.array(samples, dtype=np.int16)

    # -----------------------------------------------------------------------
    # 5. Animation update function
    # -----------------------------------------------------------------------
    def update_frame(frame):
        # Read a chunk of samples from SPI
        try:
            audio_data = read_spi_samples(CHUNK)
        except Exception as e:
            print("Error reading SPI data:", e)
            return bars

        # Apply window
        windowed_data = audio_data * window

        # Compute the FFT (positive frequencies only)
        yf = np.fft.fft(windowed_data)
        N = len(yf)
        yff = (1.0 / N) * np.abs(yf[:N // 2])

        # Optional filtering
        yff = np.array([val if val > FILTER_THRESHOLD else 0 for val in yff])

        # Prepare colors
        colors = ['r'] * (FFT_SIZE // 2)

        if display_mode == 1:
            # Single maximum mode
            if len(yff) > 1:
                max_index = yff[1:].argmax() + 1  # skip DC
                colors[max_index] = 'blue'
        else:
            # display_mode == 0 -> highlight top three peaks
            find_max = yff[1:].copy()
            highlight_colors = ['blue', 'green', 'orange']
            for i in range(min(NUM_MAXES, len(find_max))):
                idx = find_max.argmax() + 1
                colors[idx] = highlight_colors[i]
                find_max[idx - 1] = -1  # so it won't get picked again

        # Update the bars
        for bar, height, col in zip(bars, yff, colors):
            bar.set_height(height)
            bar.set_color(col)

        return bars

    # -----------------------------------------------------------------------
    # 6. Run Matplotlib animation
    # -----------------------------------------------------------------------
    ani = animation.FuncAnimation(fig, update_frame, interval=10, blit=False)
    plt.show()

    # -----------------------------------------------------------------------
    # 7. Clean up
    # -----------------------------------------------------------------------
    spi.close()

if __name__ == "__main__":
    main()
