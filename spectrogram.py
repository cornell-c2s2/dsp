#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

audioFile = "1809v2.WAV"

#samplingFreq: # samples/sec, mySound: array of amplitudes
samplingFreq, mySound = wavfile.read(audioFile)


# mySoundDataType = mySound.dtype #16 or 32 bit (ours are 16)
# signalDuration =  mySound.shape[0] / samplingFreq


#normalize to floats from [-1, 1]
mySound = mySound / (2.**15)

mySoundShape = mySound.shape
samplePoints = float(mySoundShape[0])

#if dual channel, take one
mySoundOneChannel = mySound[:,0]

#points in time distributed along sample points
timeArray = np.arange(0, samplePoints, 1)
timeArray = timeArray / samplingFreq

# #time domain (amplitude vs time)
# plt.plot(timeArray, mySoundOneChannel)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()



# compute the spectrogram; Sxx: intensity of freq at each time
frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)
intensity = np.log10(intensity) #to dB

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.show()

# Define the intensity threshold (in dB)
threshold_dB = -100  # Set your desired threshold in decibels

# Mask values below the threshold by setting them to the minimum dB value
Sxx_dB_filtered = np.where(intensity > threshold_dB, intensity, np.nan)

# # Plot the filtered spectrogram
# plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud', cmap='viridis')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.title('Spectrogram (Thresholded)')
# plt.colorbar(label='Intensity [dB]')
# plt.show()


from scipy.signal import butter, lfilter

# Redefining the band-pass filters

# Apply a Butterworth band-pass filter to isolate the peaks
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Choose the frequency range for the band-pass filter
lowcut = 2750  # Lower cutoff frequency for the peaks (adjust as necessary)
highcut = 5000  # Upper cutoff frequency for the peaks (adjust as necessary)

lowcut1 = 2750
highcut1 = 3500

lowcut2 = 3500
highcut2 = 4250

lowcut3 = 4250
highcut3 = 5000


# Step 1: Apply the band-pass filter to focus on the peaks
filtered_signal_bandpass = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)

filtered_signal_bandpass1 = butter_bandpass_filter(mySoundOneChannel, lowcut1, highcut1, samplingFreq)

filtered_signal_bandpass2 = butter_bandpass_filter(mySoundOneChannel, lowcut2, highcut2, samplingFreq)

filtered_signal_bandpass3 = butter_bandpass_filter(mySoundOneChannel, lowcut3, highcut3, samplingFreq)

# Compute the spectrogram for the final filtered signal
frequencies, times, Sxx = spectrogram(filtered_signal_bandpass, fs=samplingFreq)

# Plot the spectrogram after both band-pass and high-pass filtering
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut})')
plt.colorbar(label='Intensity [dB]')
plt.show()



frequencies, times, Sxx = spectrogram(filtered_signal_bandpass1, fs=samplingFreq)

# Plot the spectrogram after both band-pass and high-pass filtering
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram (Band-Pass {lowcut1}-{highcut1})')
plt.colorbar(label='Intensity [dB]')
plt.show()

frequencies, times, Sxx = spectrogram(filtered_signal_bandpass2, fs=samplingFreq)

# Plot the spectrogram after both band-pass and high-pass filtering
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram (Band-Pass {lowcut2}-{highcut2})')
plt.colorbar(label='Intensity [dB]')
plt.show()

frequencies, times, Sxx = spectrogram(filtered_signal_bandpass3, fs=samplingFreq)

# Plot the spectrogram after both band-pass and high-pass filtering
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram (Band-Pass {lowcut3}-{highcut3})')
plt.colorbar(label='Intensity [dB]')
plt.show()