import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft


folder = "16k"
audioFiles = os.listdir(folder)

# Set to 'True' to display graphs
for i in audioFiles:
    audioFile = folder+"/"+i
 
    # Step 1: Load the WAV file
    rate, signal = wavfile.read(audioFile)

    # Step 2: If stereo, convert to mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Optional: Normalize signal
    signal = signal / np.max(np.abs(signal))

    # Optional: Take a small segment (for clarity)
    segment = signal[:2048]

    # Step 3: Compute the real cepstrum
    spectrum = fft(segment)
    log_spectrum = np.log(np.abs(spectrum) + np.finfo(float).eps)
    cepstrum = np.real(ifft(log_spectrum))

    # Step 4: Plot the cepstrum
    plt.figure(figsize=(10, 4))
    plt.plot(cepstrum)
    plt.title("Real Cepstrum")
    plt.xlabel("Quefrency (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()