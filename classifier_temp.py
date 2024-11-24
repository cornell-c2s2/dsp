# #!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter
folder = "testing" #"audio"
audioFiles = os.listdir(folder)
showGraphsAndPrint = True
# for i in audioFiles:
#     audioFile = folder+"/"+i
audioFile = "testing/2287-sj.wav"

samplingFreq, mySound = wavfile.read(audioFile)

# Normalize
mySound = mySound / (2.**15)


# If stereo, take one channel
mySoundOneChannel = mySound[:, 0]


# np.savetxt("checkpt1.txt", mySoundOneChannel)
# print("Filtered data saved to 'checkpt1.txt'")

# frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)

# np.savetxt("original_intensities.txt", intensity)
# print("Filtered data saved to 'original_intensities.txt'")

# # Intensity to dB
# intensity = 10*np.log10(intensity/(10**-12))

# if showGraphsAndPrint: 
#     plt.figure(figsize=(10, 4))
#     plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [s]')
#     plt.title(f'Spectrogram of {i}')
#     plt.colorbar(label='Intensity [dB]')
#     plt.ylim(0, 20000)
#     plt.show()
    
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    # print("b: ", lowcut , ", " , highcut , ": " , b)
    # print("a: ", lowcut , ", " , highcut , ": " , a)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def normalize_intensity(intensity_dB):
    min_intensity = np.nanmin(intensity_dB)
    max_intensity = np.nanmax(intensity_dB)
    return (intensity_dB - min_intensity) / (max_intensity - min_intensity)

def find_midpoints():
    lower_threshold_dB = 45
    filtered_signal = butter_bandpass_filter(mySoundOneChannel, 2000, 6000, samplingFreq)
    _, times, intensity = spectrogram(filtered_signal, fs=samplingFreq)
    intensity = 10 * np.log10(intensity / (10**-12))
    intensity_dB_filtered = np.where(intensity > lower_threshold_dB, intensity, np.nan)

    blob_times = []

    for t_idx in range(intensity_dB_filtered.shape[1]):
        if np.any(~np.isnan(intensity_dB_filtered[:, t_idx])):
            blob_times.append(times[t_idx])

    cluster_midpoints = []
    current_cluster = [blob_times[0]] if blob_times else []

    # Tolerance for clustering times (in seconds)
    time_tolerance = 0.05
    # Minimum blob length in seconds
    min_blob_duration = 0.15  

    for j in range(1, len(blob_times)):
        if blob_times[j] - blob_times[j - 1] <= time_tolerance:
            current_cluster.append(blob_times[j])
        else:
            if current_cluster and (current_cluster[-1] - current_cluster[0] >= min_blob_duration):
                midpoint = sum(current_cluster) / len(current_cluster)
                cluster_midpoints.append(midpoint)
            current_cluster = [blob_times[j]]

    if current_cluster and (current_cluster[-1] - current_cluster[0] >= min_blob_duration):
        midpoint = sum(current_cluster) / len(current_cluster)
        cluster_midpoints.append(midpoint)

    # if showGraphsAndPrint:
    #     print(f"Detected blob midpoints in {i}:")
    #     for midpoint in cluster_midpoints:
    #         print(f"  Midpoint time: {midpoint:.2f} s")
    return cluster_midpoints


# Calculate Spectrogram 
lowcut = 6000
highcut = 15000
lower_threshold_dB_normalized = 0.85
upper_threshold_dB_normalized = 0.9
filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)

# Save filtered data
np.savetxt("checkpt2.txt", filtered_signal)
print("Filtered data saved to 'checkpt2.txt'")


frequencies, times, intensity = spectrogram(filtered_signal, fs=samplingFreq)
# Save filtered data
np.savetxt("intensities.txt", intensity)
print("intensities data saved to 'intensities.txt'")

np.savetxt("freqs.txt", frequencies)
print("freqs data saved to 'freqs.txt'")

from scipy import signal

def spect2(x, fs):
    """
    Compute the spectrogram of a signal using default parameters.

    Parameters
    ----------
    x : array_like
        Time series of measurement values.
    fs : float
        Sampling frequency of the `x` time series.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    times : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x.
    """
    # Default parameters
    nperseg = 256  # Length of each segment
    noverlap = nperseg // 8  # 32 points overlap
    nfft = nperseg  # FFT length
    nstep = nperseg - noverlap  # Step size between segments
    window = ('tukey', 0.25)  # Tukey window with alpha=0.25

    # Generate the window function
    win = signal.get_window(window, nperseg).astype(np.float64)

    # # Save filtered data
    # np.savetxt("winds.txt", win)
    # print("winds data saved to 'winds.txt'")

    # Compute the scale factor
    scale = 1.0 / (fs * np.sum(win ** 2))
    # print(fs * np.sum(win ** 2))

    # Determine the number of segments
    n_segments = (len(x) - nperseg) // nstep + 1

    # Prepare frequency and time arrays
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    num_freqs = nfft // 2 + 1
    times = np.arange(n_segments) * nstep / fs + (nperseg / 2) / fs

    # Initialize the spectrogram array
    Sxx = np.zeros((num_freqs, n_segments), dtype=np.float64)
    # print(n_segments)
    # Loop over each segment
    # for i in range(n_segments):
    for i in range(1):

        start = i * nstep
        segment = x[start:start + nperseg]

     
        if(i == 0):
            np.savetxt("segs1.txt", segment)
            print("segs1 data saved to 'segs1.txt'")



        # Detrend the segment (remove the mean)
        segment = segment - np.mean(segment)

       # Save filtered data
        if(i == 0):
            np.savetxt("segs2.txt", segment)
            print("mean data saved to 'segs2.txt'")
            np.savetxt("window.txt", win)
            print("window data saved to 'window.txt'")
           


        # Apply the window to the segment
        segment = segment * win


        # Save filtered data
        if(i == 0):
            np.savetxt("win.txt", segment)
            print("win data saved to 'win.txt'")

        # Compute the FFT of the segment
        fft_segment = np.fft.rfft(segment, n=nfft)
         # Save filtered data
        # np.savetxt("fftseg.txt", fft_segment)
        # print("fftseg data saved to 'fftseg.txt'")

        # Compute the power spectral density
        Sxx[:, i] = np.abs(fft_segment) ** 2 * scale

    # Adjust scaling for one-sided spectrum
    if nfft % 2 == 0:
        Sxx[1:-1, :] *= 2
    else:
        Sxx[1:, :] *= 2

    return freqs, times, Sxx

f2, t2, i2 = spect2(filtered_signal, fs=samplingFreq)
print(filtered_signal)
# Save filtered data
np.savetxt("i2.txt", i2)
print("i2 data saved to 'i2.txt'")

np.savetxt("f2.txt", f2)
print("f2 data saved to 'f2.txt'")




intensity = 10 * np.log10(intensity/(10**-12))
intensity_dB_normalized = normalize_intensity(intensity)


intensity_dB_filtered = np.where(intensity_dB_normalized > lower_threshold_dB_normalized, intensity_dB_normalized, np.nan)
intensity_dB_filtered = np.where(intensity_dB_filtered < upper_threshold_dB_normalized, intensity_dB_normalized, np.nan)





# if showGraphsAndPrint:
#     plt.figure(figsize=(10, 4))
#     plt.pcolormesh(times, frequencies, intensity_dB_filtered, shading='gouraud')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [s]')
#     plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
#     plt.colorbar(label='Intensity (Normalized) [dB]')
#     plt.ylim(0, 20000)
#     plt.show()

# Scrub Jay Classify
midpoints = find_midpoints()

has_a_scrub = False
for midpoint in midpoints:
    time_threshold = 0.18
    times_filtered = np.where((times < midpoint+time_threshold) & (times > midpoint - time_threshold), times, np.nan)

    finite_indices = np.isfinite(times_filtered)
    times_finite = times_filtered[finite_indices]
    intensity_dB_filtered_finite = intensity_dB_filtered[:, finite_indices]

    def sum_intense(lower, upper, half_range):
        freq_min_idx = np.searchsorted(frequencies, lower, side='left')
        freq_max_idx = np.searchsorted(frequencies, upper, side='right')

        time_min_idx = np.searchsorted(times, midpoint-half_range, side='left')
        time_max_idx = np.searchsorted(times, midpoint+half_range, side='right')

        area_intensity = intensity_dB_filtered[freq_min_idx:freq_max_idx, time_min_idx:time_max_idx]
        total_intensity = np.nansum(area_intensity)
        return total_intensity
    
    if showGraphsAndPrint:
        # print(i)
        print("Above: "+str(sum_intense(9000, 15000, .18)))
        print("Middle: "+str(sum_intense(7000, 8000, 0.05)))
        print("Below: "+str(sum_intense(1000, 6000, .18)))
        print()
        # plt.figure(figsize=(10, 4))
        # plt.pcolormesh(times_finite, frequencies, intensity_dB_filtered_finite, shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [s]')
        # plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
        # plt.colorbar(label='Intensity (Normalized) [dB]')
        # plt.ylim(0, 20000)
        # plt.show()

    if sum_intense(7000, 8000, 0.05) < 50 and sum_intense(9000, 15000, 0.18) > 200 and sum_intense(1000, 6000, 0.18) > 200:
        has_a_scrub = True
if has_a_scrub:
    print(audioFile + " has a Scrub Jay! :)")
else:
    print(audioFile + " has no Scrub Jay! :(")

        

