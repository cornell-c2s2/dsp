"""
Live FFT from SPIDriver using Matplotlib

Press the 'd' key in the plot window to toggle between two display modes:
    - Mode 0: Top three frequency peaks are highlighted.
    - Mode 1: Only the single maximum frequency is highlighted.

NOTE:
- This template shows how to replace spidev (or PyAudio) with spidriver.
- You must adapt SPI commands, data parsing, and timing to your actual device.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# spidriver import
from spidriver import SPIDriver

# 1) Configure how many samples to read per frame and the "audio" rate
CHUNK = 1024            # Number of samples to read per frame
RATE = 44100            # Sampling rate in Hz (if your ADC is sampling at 44.1 kHz)
                        # Adjust to match your actual hardware's sampling rate!

# 2) SPIDriver configuration
PORT_NAME = "/dev/ttyUSB0"  # Linux example. Windows might be "COM3", macOS "cu.usbserial-xxx"
SPI_SPEED = 2_000_000       # SPI bus speed (Hz). Adjust for your hardware and cable length.

# 3) FFT and plotting settings
FFT_SIZE = CHUNK
FILTER_THRESHOLD = 2
NUM_MAXES = 3

# Global for display mode
display_mode = 0  # 0: highlight top 3 peaks; 1: highlight only the maximum peak

def on_key_press(event):
    """
    Toggle display mode when 'd' key is pressed.
    """
    global display_mode
    if event.key == 'd':
        display_mode = 1 - display_mode
        print(f"Display mode changed to {display_mode}.")

def main():
    global display_mode

    # -----------------------------------------------------------------------
    # 1. Open SPIDriver
    # -----------------------------------------------------------------------
    print(f"Opening SPIDriver on port: {PORT_NAME}")
    d = SPIDriver(PORT_NAME)
    d.setspeed(SPI_SPEED)
    print(f"SPI speed set to {SPI_SPEED} Hz")

    # Depending on your device, you might need to configure mode, pins, etc.
    # For example, set CS pin high/low manually or pull other pins as needed:
    # d.setpins(d.PIN_CS)   # example usage: keep CS high if needed
    # d.setpins(0)         # example usage: set all pins low

    # -----------------------------------------------------------------------
    # 2. Create a Hanning window (helps reduce FFT spectral leakage)
    # -----------------------------------------------------------------------
    window = np.hanning(FFT_SIZE)

    # -----------------------------------------------------------------------
    # 3. Set up Matplotlib for plotting
    # -----------------------------------------------------------------------
    freqs = np.linspace(0, RATE / 2, FFT_SIZE // 2)
    fig, ax = plt.subplots()
    bars = ax.bar(freqs, np.zeros(FFT_SIZE // 2), width=freqs[1] - freqs[0], color='r')
    ax.set_xlim(0, RATE / 2)
    ax.set_ylim(0, 100)  # Adjust to match your expected FFT amplitude range
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live FFT from SPIDriver")

    # Connect key-press event for toggling display modes
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -----------------------------------------------------------------------
    # 4. Function to read samples from SPIDriver
    # -----------------------------------------------------------------------
    def read_spi_samples(num_samples):
        """
        Reads num_samples of 16-bit data via SPIDriver.
        Returns a NumPy array of signed 16-bit integers.

        NOTE:
        - This code assumes your device returns raw 16-bit samples immediately
          if we clock out zeros. In practice, you may need a custom command
          sequence or read registers from your ADC/device.
        - Adjust for endianness, bit depth, or protocol specifics.
        """
        # Each sample is 2 bytes (16 bits)
        bytes_to_read = num_samples * 2
        
        # Pull CS low before communication
        d.sel()
        
        # Write zero bytes and simultaneously read the response
        # This returns a list of integers (0..255)
        raw_data = d.write([0x00] * bytes_to_read)
        
        # Release CS
        d.unsel()
        
        # Convert returned list to a bytes object
        byte_str = bytes(raw_data)
        
        # Unpack as little-endian 16-bit integers: '<h'
        samples = struct.unpack('<' + 'h' * num_samples, byte_str)
        
        return np.array(samples, dtype=np.int16)

    # -----------------------------------------------------------------------
    # 5. Matplotlib animation function
    # -----------------------------------------------------------------------
    def update_frame(frame):
        # Read a chunk of data from the SPI
        try:
            audio_data = read_spi_samples(CHUNK)
        except Exception as e:
            print("Error reading SPI data:", e)
            return bars

        # Apply the window
        windowed_data = audio_data * window

        # Compute the FFT (only positive frequencies)
        yf = np.fft.fft(windowed_data)
        N = len(yf)
        yff = (1.0 / N) * np.abs(yf[:N // 2])

        # Optional filtering: zero out values below threshold
        yff = np.array([val if val > FILTER_THRESHOLD else 0 for val in yff])

        # Prepare default colors (red)
        colors = ['r'] * (FFT_SIZE // 2)

        if display_mode == 1:
            # Highlight only the single maximum peak (ignoring the DC component)
            if len(yff) > 1:
                max_index = yff[1:].argmax() + 1
                colors[max_index] = 'blue'
        else:
            # Highlight the top three peaks
            find_max = yff[1:].copy()
            highlight_colors = ['blue', 'green', 'orange']
            for i in range(min(NUM_MAXES, len(find_max))):
                idx = find_max.argmax() + 1
                colors[idx] = highlight_colors[i]
                find_max[idx - 1] = -1  # so it won't get picked again

        # Update the bar heights and colors
        for bar, height, col in zip(bars, yff, colors):
            bar.set_height(height)
            bar.set_color(col)

        return bars

    # -----------------------------------------------------------------------
    # 6. Start Matplotlib animation
    # -----------------------------------------------------------------------
    ani = animation.FuncAnimation(fig, update_frame, interval=10, blit=False)
    plt.show()

    # -----------------------------------------------------------------------
    # 7. Cleanup
    # -----------------------------------------------------------------------
    # Optionally, set all pins to default state, close serial, etc.
    d.close()
    print("SPIDriver closed.")

if __name__ == "__main__":
    main()
