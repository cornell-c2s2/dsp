# #!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter
# audioFiles = [ "1389.WAV"]
audioFiles = os.listdir("audio")
doFilter = True 
showGraphsAndPrint = False
for i in audioFiles:
    audioFile = "audio/"+i

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
    if showGraphsAndPrint: 
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(times, frequencies, intensity, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f'Spectrogram of {i}')
        plt.colorbar(label='Intensity [dB]')
        plt.ylim(0, 20000)
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
        
        def normalize_intensity(intensity_dB):
            min_intensity = np.nanmin(intensity_dB)
            max_intensity = np.nanmax(intensity_dB)
            # Normalize to a 0-1 scale
            return (intensity_dB - min_intensity) / (max_intensity - min_intensity)

        #Find midpoints START
        def find_midpoints():
            lower_threshold_dB = 45
            upper_threshold_dB = 1000
            filtered_signal = butter_bandpass_filter(mySoundOneChannel, 2000, 6000, samplingFreq)
            frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
            Sxx = 10 * np.log10(Sxx / (10**-12))
            Sxx_dB_filtered = np.where((Sxx > lower_threshold_dB) & (Sxx < upper_threshold_dB), Sxx, np.nan)

            # Initialize a list to store times where blobs occur
            blob_times = []

            # Loop through each time slice to detect blobs
            for t_idx in range(Sxx_dB_filtered.shape[1]):
                # Check if there are any non-NaN values in this time slice
                if np.any(~np.isnan(Sxx_dB_filtered[:, t_idx])):
                    blob_times.append(times[t_idx])

            # Detect clusters, calculate midpoints, and filter out short blobs
            cluster_midpoints = []
            current_cluster = [blob_times[0]] if blob_times else []

            # Tolerance for clustering times (in seconds)
            time_tolerance = 0.05
            min_blob_duration = 0.15  # Minimum blob length in seconds

            for j in range(1, len(blob_times)):
                if blob_times[j] - blob_times[j - 1] <= time_tolerance:
                    # Continue the current cluster
                    current_cluster.append(blob_times[j])
                else:
                    # Finalize the current cluster and calculate its midpoint if it's long enough
                    if current_cluster and (current_cluster[-1] - current_cluster[0] >= min_blob_duration):
                        midpoint = sum(current_cluster) / len(current_cluster)
                        cluster_midpoints.append(midpoint)
                    # Start a new cluster
                    current_cluster = [blob_times[j]]

            # Add the last cluster's midpoint if it exists and is long enough
            if current_cluster and (current_cluster[-1] - current_cluster[0] >= min_blob_duration):
                midpoint = sum(current_cluster) / len(current_cluster)
                cluster_midpoints.append(midpoint)

            # Print the detected cluster midpoints
            if showGraphsAndPrint:
                print(f"Detected blob midpoints in {i}:")
                for midpoint in cluster_midpoints:
                    print(f"  Midpoint time: {midpoint:.2f} s")
            return cluster_midpoints
        midpoints = find_midpoints()
        # Frequency ranges for filters
        bands = [(6000, 15000)]#, (1000, 2000), (2000, 3500), (3500, 4500)]
        lower_threshold_dB = 0.85
        upper_threshold_dB = 0.9
        for lowcut, highcut in bands:
            filtered_signal = butter_bandpass_filter(mySoundOneChannel, lowcut, highcut, samplingFreq)
            frequencies, times, Sxx = spectrogram(filtered_signal, fs=samplingFreq)
            Sxx = 10 * np.log10(Sxx/(10**-12))
            Sxx_dB_normalized = normalize_intensity(Sxx)
            Sxx_dB_filtered = np.where(Sxx_dB_normalized > lower_threshold_dB, Sxx_dB_normalized, np.nan)
            Sxx_dB_filtered = np.where(Sxx_dB_filtered < upper_threshold_dB, Sxx_dB_normalized, np.nan)
            if showGraphsAndPrint:
                plt.figure(figsize=(10, 4))
                plt.pcolormesh(times, frequencies, Sxx_dB_filtered, shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [s]')
                plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
                plt.colorbar(label='Intensity [dB]')
                plt.ylim(0, 20000)
                plt.show()

            has_a_scrub = False
            for midpoint in midpoints:
                time_threshold = 0.18
                times_filtered = np.where((times < midpoint+time_threshold) & (times > midpoint - time_threshold), times, np.nan)

                finite_indices = np.isfinite(times_filtered)
                times_finite = times_filtered[finite_indices]  # Only non-NaN times
                Sxx_dB_filtered_finite = Sxx_dB_filtered[:, finite_indices]  # Match dimensions with times_finite

                def sum_intense(lower, upper, half_range):
                    freq_min_idx = np.searchsorted(frequencies, lower, side='left')
                    freq_max_idx = np.searchsorted(frequencies, upper, side='right')

                    # Find time indices within the specified range
                    time_min_idx = np.searchsorted(times, midpoint-half_range, side='left')
                    time_max_idx = np.searchsorted(times, midpoint+half_range, side='right')

                    # Slice the Sxx matrix to get the specified area
                    area_intensity = Sxx_dB_filtered[freq_min_idx:freq_max_idx, time_min_idx:time_max_idx]
                    #print(area_intensity)
                    # Sum the intensities in the specified area, ignoring NaNs
                    total_intensity = np.nansum(area_intensity)
                    return total_intensity
                if showGraphsAndPrint:
                    print("Above: "+str(sum_intense(9000, 15000, .18)))
                    print("Middle: "+str(sum_intense(7000, 8000, 0.05)))
                    print("Below: "+str(sum_intense(1000, 6000, .18)))
                    print()
                    # Plot the spectrogram with filtered values
                    plt.figure(figsize=(10, 4))
                    plt.pcolormesh(times_finite, frequencies, Sxx_dB_filtered_finite, shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [s]')
                    plt.title(f'Spectrogram (Band-Pass {lowcut}-{highcut}) of {i}')
                    plt.colorbar(label='Intensity [dB]')
                    plt.ylim(0, 20000)
                    plt.show()

                if sum_intense(7000, 8000, 0.05) < 50 and sum_intense(9000, 15000, .18) > 200 and sum_intense(1000, 6000, .18) > 200:
                    has_a_scrub = True
            if has_a_scrub:
                print(audioFile+" has a Scrub Jay! :)")
            else:
                print(audioFile+" has no Scrub Jay! :(")

                

