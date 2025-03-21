#========================================================================
# classifier16k.py
#========================================================================
# A python implementation of the Donut Classifier with 16k sampling rate

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter

# Location of the audio files
folder = "16k-single-test"
audioFiles = os.listdir(folder)

# Set to 'True' to display graphs
showGraphsAndPrint = True
for i in audioFiles:
    audioFile = folder+"/"+i
    # Read audio file
    samplingFreq, mySound = wavfile.read(audioFile)
    # Normalize
    mySound = mySound / (2.**15)
    # If stereo, take one channel
    mySoundOneChannel = mySound[:, 0]
    # Compute spectrogram
    frequencies, times, intensity = spectrogram(mySoundOneChannel, fs=samplingFreq)
    # Intensity to dB
    intensity = 10*np.log10(intensity/(10**-12))

    if showGraphsAndPrint: 
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f'Spectrogram of {i}')
        plt.colorbar(label='Intensity [dB]')
        plt.ylim(0, 10000)
        plt.show()
        
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    def normalize_intensity(intensity_dB):
        min_intensity = np.nanmin(intensity_dB)
        max_intensity = np.nanmax(intensity_dB)
        return (intensity_dB - min_intensity) / (max_intensity - min_intensity)

    def find_midpoints():
        # Filter signal and compute spectrogram (above intensity threshold)
        lower_threshold_dB = 45
        filtered_signal = butter_bandpass_filter(mySoundOneChannel, 1000, 3000, samplingFreq)
        _, times, intensity = spectrogram(filtered_signal, fs=samplingFreq)
        intensity = 10 * np.log10(intensity / (10**-12))
        intensity_dB_filtered = np.where(intensity > lower_threshold_dB, intensity, np.nan)

        blob_times = []

        for t_idx in range(intensity_dB_filtered.shape[1]):
            if np.any(~np.isnan(intensity_dB_filtered[:, t_idx])):
                blob_times.append(times[t_idx])

        cluster_midpoints = []
        current_cluster = [blob_times[0]] if blob_times else []

        # Tolerance for clustering times in seconds
        time_tolerance = 0.05
        # Minimum blob length in seconds
        min_blob_duration = 0.15  

        # Find valid midpoints
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

        if showGraphsAndPrint:
            print(f"Detected blob midpoints in {i}:")
            for midpoint in cluster_midpoints:
                print(f"  Midpoint time: {midpoint:.2f} s")
        return cluster_midpoints
    midpoints = find_midpoints()

    # Calculate Spectrogram 
    lowcut = 3000
    highcut = 7500
    lower_threshold_dB_normalized = 0.70
    upper_threshold_dB_normalized = 0.85
    filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)

    impulse = np.zeros(100)
    impulse[0] = 1

    # Apply Butterworth bandpass filter
    filtered_impulse = butter_bandpass_filter(impulse, lowcut, highcut, samplingFreq)

    frequencies, times, intensity = spectrogram(filtered_signal, fs=samplingFreq)

    # Normalize and filter intensities 
    intensity = 10 * np.log10(intensity/(10**-12))
    intensity_dB_normalized = normalize_intensity(intensity)
    intensity_dB_filtered = np.where(intensity_dB_normalized > lower_threshold_dB_normalized, intensity_dB_normalized, np.nan)
    intensity_dB_filtered = np.where(intensity_dB_filtered < upper_threshold_dB_normalized, intensity_dB_normalized, np.nan)

    if showGraphsAndPrint:
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, intensity_dB_filtered, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
        plt.colorbar(label='Intensity (Normalized) [dB]')
        plt.ylim(0, 10000)
        plt.show()

    # Scrub Jay Classify
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
        
        print("Above: "+str(sum_intense(5000, 7000, .18)))
        print("Middle: "+str(sum_intense(2500, 5000, 0.05)))
        print("Below: "+str(sum_intense(500, 2500, .18)))
        print()
        if showGraphsAndPrint:
            plt.figure(figsize=(10, 4))
            plt.pcolormesh(times_finite, frequencies, intensity_dB_filtered_finite, shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
            plt.colorbar(label='Intensity (Normalized) [dB]')
            plt.ylim(0, 10000)
            plt.show()
        
        # Check thresholds
        if sum_intense(2500, 5000, 0.05) < 75 and sum_intense(5000, 7000, 0.18) > 300 and sum_intense(500, 2500, 0.18) > 100:
            has_a_scrub = True
    if has_a_scrub:
        print(audioFile + " has a Scrub Jay! :)")
    else:
        print(audioFile + " has no Scrub Jay! :(")

        

