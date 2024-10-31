# #!/usr/bin/env python

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from scipy.signal import spectrogram, butter, lfilter
# # audioFiles = [ "1389.WAV"]
# audioFiles = os.listdir("audio")
# doFilter = True 
# for i in audioFiles:
#     audioFile = "audio/"+i

#     # Load audio file
#     samplingFreq, mySound = wavfile.read(audioFile)

#     # Normalize audio signal
#     mySound = mySound / (2.**15)

#     # If stereo, take one channel
#     mySoundOneChannel = mySound[:, 0]

#     # Compute the spectrogram; convert intensity to dB
#     frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)
#     intensity = 10*np.log10(intensity/(10**-12))
#     #Plot Spectrogram
#     plt.figure(figsize=(10, 4))
#     plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [s]')
#     plt.title(f'Spectrogram of {i}')
#     plt.colorbar(label='Intensity [dB]')
#     plt.ylim(0, 15000)
#     plt.show()
#     if doFilter:
#         # Define Butterworth band-pass filter
#         def butter_bandpass(lowcut, highcut, fs, order=4):
#             nyquist = 0.5 * fs
#             low = lowcut / nyquist
#             high = highcut / nyquist
#             b, a = butter(order, [low, high], btype='band')
#             return b, a

#         def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#             b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#             return lfilter(b, a, data)

#         # Frequency ranges for filters
#         bands = [(1000, 5000)]#, (1000, 2000), (2000, 3500), (3500, 4500)]
#         lower_threshold_dB = 20
#         upper_threshold_dB = 45


#         for lowcut, highcut in bands:
#             filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)
#             frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
#             Sxx = 10 * np.log10(Sxx/(10**-12))
#             Sxx_dB_filtered = np.where(Sxx > lower_threshold_dB, Sxx, np.nan)
#             Sxx_dB_filtered = np.where(Sxx_dB_filtered < upper_threshold_dB, Sxx, np.nan)
            
#             plt.figure(figsize=(10, 4))
#             plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud')
#             plt.ylabel('Frequency [Hz]')
#             plt.xlabel('Time [s]')
#             plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
#             plt.colorbar(label='Intensity [dB]')
#             plt.ylim(0, 15000)
#             plt.show()



#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter

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

# Directory for audio files
audioFiles = os.listdir("audio")
doFilter = True 
time_interval = 0.1  # Time interval in seconds for intensity analysis
samplingFreq = 44100  # Replace with actual sampling frequency if known

# Frequency bands and intensity thresholds
frequency_bands = [(1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000),
                   (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000)]
lower_threshold_dB = 20
upper_threshold_dB = 45
intensity_change_threshold = 10  # Define an intensity change threshold to detect shape

for i in audioFiles:
    audioFile = "audio/" + i

    # Load and normalize audio file
    samplingFreq, mySound = wavfile.read(audioFile)
    mySound = mySound / (2.**15)
    mySoundOneChannel = mySound[:, 0] if mySound.ndim > 1 else mySound

    # Process each frequency band
    for lowcut, highcut in frequency_bands:
        # Filter the signal for the current frequency band
        filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)

        # Compute the spectrogram of the filtered signal
        frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
        Sxx_dB = 10 * np.log10(Sxx / (10**-12))

        # Apply intensity threshold
        Sxx_dB_filtered = np.where((Sxx_dB > lower_threshold_dB) & (Sxx_dB < upper_threshold_dB), Sxx_dB, np.nan)

        # Plot the filtered spectrogram
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut} Hz) of {i}')
        plt.colorbar(label='Intensity [dB]')
        plt.ylim(lowcut, highcut)
        plt.show()

        # Step 3: Divide into intervals and calculate intensity sum
        interval_size = int(time_interval * samplingFreq)
        num_intervals = len(times) // interval_size
        intensity_sums = []

        for interval in range(num_intervals - 1):
            interval_start = interval * interval_size
            interval_end = interval_start + interval_size
            intensity_sum = np.nansum(Sxx_dB_filtered[:, interval_start:interval_end], axis=1)
            intensity_sums.append(intensity_sum)

        # Step 4: Detect significant changes in intensity
        for idx in range(1, len(intensity_sums)):
            intensity_change = intensity_sums[idx] - intensity_sums[idx - 1]
            if np.max(intensity_change) > intensity_change_threshold:
                print(f"Potential shape detected in band {lowcut}-{highcut} Hz at time interval {idx * time_interval} s")

