#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

audioFile = "iir/1363v2.WAV"

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

# # Plot the spectrogram
# plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.title('Spectrogram')
# plt.colorbar(label='Intensity [dB]')
# plt.show()

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

# Redefining the band-pass and high-pass filters

# Apply a Butterworth band-pass filter to isolate the peaks
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Apply a Butterworth high-pass filter to remove low-frequency noise
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

# Choose the frequency range for the band-pass filter
lowcut = 1000  # Lower cutoff frequency for the peaks (adjust as necessary)
highcut = 5000  # Upper cutoff frequency for the peaks (adjust as necessary)

# Choose the cutoff frequency for the high-pass filter
highpass_cutoff = 300  # Set the high-pass cutoff to remove low-frequency noise (e.g., 300 Hz)

# Step 1: Apply the band-pass filter to focus on the peaks
filtered_signal_bandpass = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)

# Step 2: Apply the high-pass filter to remove remaining low-frequency noise
final_filtered_signal = butter_highpass_filter(filtered_signal_bandpass, highpass_cutoff, samplingFreq)

# Compute the spectrogram for the final filtered signal
frequencies, times, Sxx = spectrogram(final_filtered_signal, fs=samplingFreq)

# Plot the spectrogram after both band-pass and high-pass filtering
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut} Hz + High-Pass {highpass_cutoff} Hz)')
plt.colorbar(label='Intensity [dB]')
plt.show()



# # Design a Butterworth IIR filter (low-pass example)
# def butter_lowpass(cutoff, fs, order=5):
#     nyquist = 0.5 * fs  # Nyquist frequency is half of the sampling rate
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# # Apply the IIR filter to the signal
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# # Apply a low-pass filter with a cutoff frequency of 1000 Hz
# cutoff_frequency = 1000  # Example cutoff frequency in Hz
# filtered_signal = butter_lowpass_filter(mySoundOneChannel, cutoff_frequency, samplingFreq)

# # Now compute and plot the spectrogram for the filtered signal
# frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)



# # Design a Butterworth high-pass filter
# def butter_highpass(cutoff, fs, order=5):
#     nyquist = 0.5 * fs  # Nyquist frequency
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return b, a

# # Apply the Butterworth high-pass filter
# def butter_highpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_highpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# # Apply a high-pass filter with a cutoff frequency of around 200 Hz (adjust as needed)
# highpass_cutoff = 5000  # Cutoff frequency in Hz to remove low-frequency noise
# filtered_signal_highpass = butter_highpass_filter(mySoundOneChannel, highpass_cutoff, samplingFreq)

# # Compute the spectrogram for the high-pass filtered signal
# frequencies, times, Sxx = spectrogram(filtered_signal_highpass, fs=samplingFreq)

# # Plot the spectrogram after high-pass filtering
# plt.figure(figsize=(10, 4))
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.title('Spectrogram of High-Pass Filtered Signal (Cutoff at 8000 Hz)')
# plt.colorbar(label='Intensity [dB]')
# plt.show()



# # plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gour aud')
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [s]')
# # plt.title('Spectrogram of Filtered Signal')
# # plt.colorbar(label='Intensity [dB]')
# # plt.show()
