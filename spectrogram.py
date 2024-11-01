# #!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter
# audioFiles = [ "1389.WAV"]
audioFiles = os.listdir("bit_test")
doFilter = True 
for i in audioFiles:
    audioFile = "bit_test/"+i

    # Load audio file
    samplingFreq, mySound = wavfile.read(audioFile)

    # Normalize audio signal
    mySound = mySound / (2.**15)

    # If stereo, take one channel
    mySoundOneChannel = mySound[:, 0]

    # Compute the spectrogram; convert intensity to dB
    frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)
    intensity = 10*np.log10(intensity/(10**-12))
    #Plot Spectrogram
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram of {i}')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, 15000)
    plt.show()
    if doFilter:
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
        bands = [(1000, 5000)]#, (1000, 2000), (2000, 3500), (3500, 4500)]
        lower_threshold_dB = 20
        upper_threshold_dB = 45


        for lowcut, highcut in bands:
            filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)
            frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
            Sxx = 10 * np.log10(Sxx/(10**-12))
            Sxx_dB_filtered = np.where(Sxx > lower_threshold_dB, Sxx, np.nan)
            Sxx_dB_filtered = np.where(Sxx_dB_filtered < upper_threshold_dB, Sxx, np.nan)
            
            plt.figure(figsize=(10, 4))
            plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
            plt.colorbar(label='Intensity [dB]')
            plt.ylim(0, 15000)
            plt.show()

