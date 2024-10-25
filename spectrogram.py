#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter

audioFile = "1809v2.WAV"

# Load audio file
samplingFreq, mySound = wavfile.read(audioFile)

# Normalize audio signal
mySound = mySound / (2.**15)

# If stereo, take one channel
mySoundOneChannel = mySound[:, 0]

# Compute the spectrogram; convert intensity to dB
frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)
intensity = np.log10(intensity)

# Plot the spectrogram with frequency limit up to 10,000 Hz
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.ylim(0, 15000)  # Set frequency limit to 10,000 Hz
plt.show()

# Define Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Frequency ranges for filters
bands = [(2750, 5000), (2750, 3500), (3500, 4250), (4250, 5000)]
threshold_dB = -100

for lowcut, highcut in bands:
    filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)
    frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
    Sxx = 10 * np.log10(Sxx)
    Sxx_dB_filtered = np.where(Sxx > threshold_dB, Sxx, np.nan)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut})')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, 15000)  # Set frequency limit to 10,000 Hz
    plt.show()
