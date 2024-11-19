#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sndfile.h>
#include <fftw3.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <complex.h>
#include <dirent.h>

// Define constants
#define MAX_FILENAME 256
#define PI 3.14159265358979323846

// Function prototypes
bool butter_bandpass(double lowcut, double highcut, double fs, int order, double *b, double *a);
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output);
void compute_spectrogram(double *signal, int signal_length, int fs, double **frequencies, double **times, double ***intensity, int *freq_bins, int *time_bins);
double normalize_intensity(double value, double min, double max);
void find_midpoints(double *signal, int signal_length, int fs, double *frequencies, double *times, double **intensity, int freq_bins, int time_bins, double lower_threshold_dB, double min_blob_duration, double time_tolerance, double *midpoints, int *midpoint_count);
double sum_intense(double *intensity, int freq_bins, int time_bins, double *frequencies, double *times, double midpoint, double lower, double upper, double half_range);

int main()
{
    // Variables
    char folder[] = "testing";
    struct dirent *entry;
    DIR *directory = opendir(folder);
    while ((entry = readdir(directory)) != NULL)
    {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        {
            char audioFile[MAX_FILENAME];

            snprintf(audioFile, sizeof(audioFile), "testing/%s", entry->d_name);
            bool showGraphsAndPrint = false; // Set to true to enable plotting and printing

            // Read audio file using libsndfile
            SF_INFO sfinfo;
            SNDFILE *infile = sf_open(audioFile, SFM_READ, &sfinfo);
            if (!infile)
            {
                printf("Could not open audio file %s\n", audioFile);
                return 1;
            }

            int num_channels = sfinfo.channels;
            int samplingFreq = sfinfo.samplerate;
            int num_frames = sfinfo.frames;
            double *mySound = (double *)malloc(num_frames * num_channels * sizeof(double));

            // Read the samples as double
            sf_count_t num_read = sf_read_double(infile, mySound, num_frames * num_channels);
            if (num_read != num_frames * num_channels)
            {
                printf("Failed to read all samples from %s\n", audioFile);
                sf_close(infile);
                free(mySound);
                return 1;
            }
            sf_close(infile);

            // Normalize the audio signal
            for (int i = 0; i < num_frames * num_channels; i++)
            {
                mySound[i] /= 32768.0; // Assuming 16-bit PCM
            }

            // If stereo, take one channel
            double *mySoundOneChannel = (double *)malloc(num_frames * sizeof(double));
            if (num_channels == 2)
            {
                for (int i = 0; i < num_frames; i++)
                {
                    mySoundOneChannel[i] = mySound[2 * i]; // Take the first channel
                }
            }
            else
            {
                for (int i = 0; i < num_frames; i++)
                {
                    mySoundOneChannel[i] = mySound[i];
                }
            }
            free(mySound);

            // Compute spectrogram (simplified)
            double *frequencies = NULL;
            double *times = NULL;
            double **intensity = NULL;
            int freq_bins = 0, time_bins = 0;

            compute_spectrogram(mySoundOneChannel, num_frames, samplingFreq, &frequencies, &times, &intensity, &freq_bins, &time_bins);

            // Convert intensity to dB
            for (int i = 0; i < freq_bins; i++)
            {
                for (int j = 0; j < time_bins; j++)
                {
                    if (intensity[i][j] > 0)
                        intensity[i][j] = 10 * log10(intensity[i][j] / 1e-12);
                    else
                        intensity[i][j] = -INFINITY;
                }
            }

            // Optionally plot the spectrogram
            if (showGraphsAndPrint)
            {
                // Plotting in C is non-trivial. You can output the data to a file and use Gnuplot to visualize.
                FILE *fp = fopen("spectrogram_data.txt", "w");
                for (int i = 0; i < freq_bins; i++)
                {
                    for (int j = 0; j < time_bins; j++)
                    {
                        fprintf(fp, "%f %f %f\n", times[j], frequencies[i], intensity[i][j]);
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                // Command to plot using Gnuplot (execute separately)
                printf("Spectrogram data saved to 'spectrogram_data.txt'. Use Gnuplot to visualize.\n");
            }

            // Define Butterworth bandpass filter parameters
            double lowcut = 2000.0;
            double highcut = 6000.0;
            int order = 2;       // Changed to 2 for second-order filter
            double b[3] = {0.0}; // Coefficients for second-order filter
            double a[3] = {0.0};

            // Get Butterworth filter coefficients
            if (!butter_bandpass(lowcut, highcut, (double)samplingFreq, order, b, a))
            {
                printf("Failed to design Butterworth filter.\n");
                free(mySoundOneChannel);
                // Free spectrogram data
                for (int i = 0; i < freq_bins; i++)
                {
                    free(intensity[i]);
                }
                free(intensity);
                free(frequencies);
                free(times);
                return 1;
            }

            // Apply Butterworth bandpass filter
            double *filtered_signal = (double *)malloc(num_frames * sizeof(double));
            butter_bandpass_filter(mySoundOneChannel, num_frames, b, a, filtered_signal);

            // Find midpoints
            double lower_threshold_dB = 45.0;
            double *midpoints = (double *)malloc(time_bins * sizeof(double)); // Maximum possible
            int midpoint_count = 0;

            find_midpoints(filtered_signal, num_frames, samplingFreq, frequencies, times, intensity, freq_bins, time_bins, lower_threshold_dB, 0.15, 0.05, midpoints, &midpoint_count);

            // Calculate Spectrogram with new bandpass
            lowcut = 6000.0;
            highcut = 15000.0;
            double lower_threshold_dB_normalized = 0.85;
            double upper_threshold_dB_normalized = 0.9;

            // Design new Butterworth filter
            if (!butter_bandpass(lowcut, highcut, (double)samplingFreq, order, b, a))
            {
                printf("Failed to design Butterworth filter.\n");
                free(mySoundOneChannel);
                free(filtered_signal);
                free(midpoints);
                // Free spectrogram data
                for (int i = 0; i < freq_bins; i++)
                {
                    free(intensity[i]);
                }
                free(intensity);
                free(frequencies);
                free(times);
                return 1;
            }

            // Apply Butterworth bandpass filter
            double *filtered_signal_bp = (double *)malloc(num_frames * sizeof(double));
            butter_bandpass_filter(mySoundOneChannel, num_frames, b, a, filtered_signal_bp);

            // Create impulse
            double impulse[100] = {0};
            impulse[0] = 1.0;

            // Apply Butterworth filter to impulse
            double *filtered_impulse = (double *)malloc(100 * sizeof(double));
            butter_bandpass_filter(impulse, 100, b, a, filtered_impulse);

            // Save filtered_signal_bp to a text file
            FILE *fp_filtered = fopen("filtered_data_c.txt", "w");
            for (int i = 0; i < num_frames; i++)
            {
                fprintf(fp_filtered, "%f\n", filtered_signal_bp[i]);
            }
            fclose(fp_filtered);
            printf("Filtered data saved to 'filtered_data_c.txt'\n");

            // Compute spectrogram of filtered_signal_bp
            double *frequencies_bp = NULL;
            double *times_bp = NULL;
            double **intensity_bp = NULL;
            int freq_bins_bp = 0, time_bins_bp = 0;

            compute_spectrogram(filtered_signal_bp, num_frames, samplingFreq, &frequencies_bp, &times_bp, &intensity_bp, &freq_bins_bp, &time_bins_bp);

            // Save intensity_bp to a text file
            FILE *fp_intensity = fopen("filtered_intensity_c.txt", "w");
            for (int i = 0; i < freq_bins_bp; i++)
            {
                for (int j = 0; j < time_bins_bp; j++)
                {
                    fprintf(fp_intensity, "%e ", intensity_bp[i][j]);
                }
                fprintf(fp_intensity, "\n");
            }
            fclose(fp_intensity);
            printf("Filtered intensity saved to 'filtered_intensity_c.txt'\n");

            // Convert intensity to dB and normalize
            double min_intensity = DBL_MAX;
            double max_intensity = -DBL_MAX;
            for (int i = 0; i < freq_bins_bp; i++)
            {
                for (int j = 0; j < time_bins_bp; j++)
                {
                    if (intensity_bp[i][j] > 0)
                    {
                        intensity_bp[i][j] = 10 * log10(intensity_bp[i][j] / 1e-12);
                        if (intensity_bp[i][j] < min_intensity)
                            min_intensity = intensity_bp[i][j];
                        if (intensity_bp[i][j] > max_intensity)
                            max_intensity = intensity_bp[i][j];
                    }
                    else
                    {
                        intensity_bp[i][j] = NAN;
                    }
                }
            }

            // Normalize intensity
            double **intensity_normalized = (double **)malloc(freq_bins_bp * sizeof(double *));
            for (int i = 0; i < freq_bins_bp; i++)
            {
                intensity_normalized[i] = (double *)malloc(time_bins_bp * sizeof(double));
                for (int j = 0; j < time_bins_bp; j++)
                {
                    if (!isnan(intensity_bp[i][j]))
                        intensity_normalized[i][j] = (intensity_bp[i][j] - min_intensity) / (max_intensity - min_intensity);
                    else
                        intensity_normalized[i][j] = NAN;
                }
            }

            // Apply normalized dB thresholds
            double **intensity_dB_filtered = (double **)malloc(freq_bins_bp * sizeof(double *));
            for (int i = 0; i < freq_bins_bp; i++)
            {
                intensity_dB_filtered[i] = (double *)malloc(time_bins_bp * sizeof(double));
                for (int j = 0; j < time_bins_bp; j++)
                {
                    if (intensity_normalized[i][j] > lower_threshold_dB_normalized && intensity_normalized[i][j] < upper_threshold_dB_normalized)
                        intensity_dB_filtered[i][j] = intensity_normalized[i][j];
                    else
                        intensity_dB_filtered[i][j] = NAN;
                }
            }

            // Optionally plot the normalized spectrogram
            if (showGraphsAndPrint)
            {
                FILE *fp_norm = fopen("normalized_spectrogram_data.txt", "w");
                for (int i = 0; i < freq_bins_bp; i++)
                {
                    for (int j = 0; j < time_bins_bp; j++)
                    {
                        fprintf(fp_norm, "%f %f %f\n", times_bp[j], frequencies_bp[i], intensity_dB_filtered[i][j]);
                    }
                    fprintf(fp_norm, "\n");
                }
                fclose(fp_norm);
                printf("Normalized spectrogram data saved to 'normalized_spectrogram_data.txt'. Use Gnuplot to visualize.\n");
            }

            // Scrub Jay Classification
            bool has_a_scrub = false;
            for (int m = 0; m < midpoint_count; m++)
            {
                double midpoint = midpoints[m];
                double time_threshold = 0.18;

                // Filter times within threshold
                // For simplicity, we iterate and check conditions
                // Implement sum_intense functionality

                double sum_above = sum_intense((double *)intensity_dB_filtered, freq_bins_bp, time_bins_bp, frequencies_bp, times_bp, midpoint, 9000.0, 15000.0, 0.18);
                double sum_middle = sum_intense((double *)intensity_dB_filtered, freq_bins_bp, time_bins_bp, frequencies_bp, times_bp, midpoint, 7000.0, 8000.0, 0.05);
                double sum_below = sum_intense((double *)intensity_dB_filtered, freq_bins_bp, time_bins_bp, frequencies_bp, times_bp, midpoint, 1000.0, 6000.0, 0.18);

                if (true)
                {
                    printf("Above: %f\n", sum_above);
                    printf("Middle: %f\n", sum_middle);
                    printf("Below: %f\n\n", sum_below);
                    // Plotting filtered spectrogram around the midpoint is omitted for brevity
                }

                if (sum_middle < 50.0 && sum_above > 200.0 && sum_below > 200.0)
                {
                    has_a_scrub = true;
                    break;
                }
            }

            if (has_a_scrub)
            {
                printf("%s has a Scrub Jay! :)\n", audioFile);
            }
            else
            {
                printf("%s has no Scrub Jay! :(\n", audioFile);
            }

            // Free allocated memory
            free(mySoundOneChannel);
            free(filtered_signal);
            free(filtered_signal_bp);
            free(filtered_impulse);
            free(midpoints);

            // Free spectrogram data
            for (int i = 0; i < freq_bins; i++)
            {
                free(intensity[i]);
            }
            free(intensity);
            free(frequencies);
            free(times);

            for (int i = 0; i < freq_bins_bp; i++)
            {
                free(intensity_bp[i]);
                free(intensity_normalized[i]);
                free(intensity_dB_filtered[i]);
            }
            free(intensity_bp);
            free(intensity_normalized);
            free(intensity_dB_filtered);
            free(frequencies_bp);
            free(times_bp);
        }
    }
    return 0;
}

/**
 * Designs a Butterworth bandpass filter using the bilinear transform method.
 * Supports second-order (order=2) filters.
 *
 * @param lowcut Low cut-off frequency in Hz
 * @param highcut High cut-off frequency in Hz
 * @param fs Sampling frequency in Hz
 * @param order Filter order (only supports order=2)
 * @param b Output array for numerator coefficients (size 3)
 * @param a Output array for denominator coefficients (size 3)
 * @return true on success, false on failure
 */
bool butter_bandpass(double lowcut, double highcut, double fs, int order, double *b, double *a)
{
    if (order != 2)
    {
        fprintf(stderr, "Only second-order (order=2) Butterworth filters are implemented.\n");
        return false;
    }

    // Pre-warp the frequencies
    double nyq = 0.5 * fs;
    double low = lowcut / nyq;
    double high = highcut / nyq;

    if (low <= 0 || high >= 1 || low >= high)
    {
        fprintf(stderr, "Invalid cutoff frequencies.\n");
        return false;
    }

    // Calculate bandwidth and center frequency
    double Bw = highcut - lowcut;
    double W0 = sqrt(lowcut * highcut);

    // Compute the Butterworth poles (only for order=2)
    double Q = W0 / Bw;

    // Compute intermediate variables
    double theta = 2.0 * PI * W0 / fs;
    double alpha = sin(theta) / (2.0 * Q);

    // Compute filter coefficients using the bilinear transform
    double cos_theta = cos(theta);
    double a0 = 1.0 + alpha;
    b[0] = alpha;
    b[1] = 0.0;
    b[2] = -alpha;
    a[0] = 1.0;
    a[1] = -2.0 * cos_theta;
    a[2] = 1.0 - alpha;

    // Normalize the filter coefficients
    b[0] /= a0;
    b[1] /= a0;
    b[2] /= a0;
    a[1] /= a0;
    a[2] /= a0;

    return true;
}

/**
 * Applies a Butterworth bandpass filter to the input data.
 *
 * @param data Input signal array
 * @param n Number of samples in the input signal
 * @param b Numerator coefficients (size 3)
 * @param a Denominator coefficients (size 3)
 * @param output Output signal array
 */
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output)
{
    // Initialize filter state
    double x1 = 0.0, x2 = 0.0;
    double y1 = 0.0, y2 = 0.0;

    for (int i = 0; i < n; i++)
    {
        // Direct Form I implementation
        output[i] = b[0] * data[i] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2;

        // Update states
        x2 = x1;
        x1 = data[i];
        y2 = y1;
        y1 = output[i];
    }
}

// Function to compute spectrogram (simplified)
void compute_spectrogram(double *signal, int signal_length, int fs, double **frequencies, double **times, double ***intensity, int *freq_bins, int *time_bins)
{
    // Parameters for spectrogram
    int window_size = 1024;
    int hop_size = 512;
    int nfft = 1024;

    *freq_bins = nfft / 2 + 1;
    *time_bins = (signal_length - window_size) / hop_size + 1;

    // Allocate memory
    *frequencies = (double *)malloc((*freq_bins) * sizeof(double));
    *times = (double *)malloc((*time_bins) * sizeof(double));
    *intensity = (double **)malloc((*freq_bins) * sizeof(double *));
    for (int i = 0; i < *freq_bins; i++)
    {
        (*intensity)[i] = (double *)calloc(*time_bins, sizeof(double));
    }

    // Compute frequency values
    for (int i = 0; i < *freq_bins; i++)
    {
        (*frequencies)[i] = (double)i * fs / nfft;
    }

    // Initialize FFTW
    fftw_complex *out;
    fftw_plan p;
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft);
    double *window = (double *)malloc(window_size * sizeof(double));
    // Simple Hanning window
    for (int i = 0; i < window_size; i++)
    {
        window[i] = 0.5 * (1 - cos(2 * PI * i / (window_size - 1)));
    }

    // Zero-padding to nfft
    double *window_padded = (double *)calloc(nfft, sizeof(double));

    p = fftw_plan_dft_r2c_1d(nfft, window_padded, out, FFTW_ESTIMATE);

    // Iterate over windows
    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;
        // Apply window and zero-padding
        for (int i = 0; i < window_size; i++)
        {
            if (start + i < signal_length)
                window_padded[i] = signal[start + i] * window[i];
            else
                window_padded[i] = 0.0;
        }
        for (int i = window_size; i < nfft; i++)
        {
            window_padded[i] = 0.0;
        }

        // Execute FFT
        fftw_execute(p);

        // Compute magnitude squared
        for (int f = 0; f < *freq_bins; f++)
        {
            double real = out[f][0];
            double imag = out[f][1];
            (*intensity)[f][t] += real * real + imag * imag;
        }

        // Compute time
        (*times)[t] = (double)(start + window_size / 2) / fs;
    }

    fftw_destroy_plan(p);
    fftw_free(out);
    free(window);
    free(window_padded);
}

// Function to find midpoints (simplified)
void find_midpoints(double *signal, int signal_length, int fs, double *frequencies, double *times, double **intensity, int freq_bins, int time_bins, double lower_threshold_dB, double min_blob_duration, double time_tolerance, double *midpoints, int *midpoint_count)
{
    // Apply Butterworth bandpass filter (already filtered before calling this function)

    // Find times where intensity > threshold
    bool *blob_present = (bool *)calloc(time_bins, sizeof(bool));
    for (int t = 0; t < time_bins; t++)
    {
        for (int f = 0; f < freq_bins; f++)
        {
            if (!isnan(intensity[f][t]) && intensity[f][t] > lower_threshold_dB)
            {
                blob_present[t] = true;
                break;
            }
        }
    }

    // Collect blob times
    double *blob_times = (double *)malloc(time_bins * sizeof(double));
    int blob_count = 0;
    for (int t = 0; t < time_bins; t++)
    {
        if (blob_present[t])
        {
            blob_times[blob_count++] = times[t];
        }
    }

    free(blob_present);

    // Cluster blob_times into midpoints
    double current_cluster_start = blob_count > 0 ? blob_times[0] : 0.0;
    double current_cluster_end = blob_count > 0 ? blob_times[0] : 0.0;
    int clusters = 0;

    for (int i = 1; i < blob_count; i++)
    {
        if (blob_times[i] - blob_times[i - 1] <= time_tolerance)
        {
            current_cluster_end = blob_times[i];
        }
        else
        {
            if ((current_cluster_end - current_cluster_start) >= min_blob_duration)
            {
                midpoints[*midpoint_count] = (current_cluster_start + current_cluster_end) / 2.0;
                (*midpoint_count)++;
                clusters++;
            }
            current_cluster_start = blob_times[i];
            current_cluster_end = blob_times[i];
        }
    }

    // Check last cluster
    if (blob_count > 0 && (current_cluster_end - current_cluster_start) >= min_blob_duration)
    {
        midpoints[*midpoint_count] = (current_cluster_start + current_cluster_end) / 2.0;
        (*midpoint_count)++;
        clusters++;
    }

    free(blob_times);
}

/**
 * Sums the intensity in a specific frequency and time range.
 *
 * @param intensity Flattened intensity array (freq_bins * time_bins)
 * @param freq_bins Number of frequency bins
 * @param time_bins Number of time bins
 * @param frequencies Array of frequency values
 * @param times Array of time values
 * @param midpoint Midpoint time around which to sum
 * @param lower Lower frequency bound in Hz
 * @param upper Upper frequency bound in Hz
 * @param half_range Half of the time range around the midpoint in seconds
 * @return Sum of intensities within the specified range
 */
double sum_intense(double *intensity, int freq_bins, int time_bins, double *frequencies, double *times, double midpoint, double lower, double upper, double half_range)
{
    double sum = 0.0;
    // Determine frequency indices
    int freq_min_idx = 0, freq_max_idx = freq_bins;
    for (int f = 0; f < freq_bins; f++)
    {
        if (frequencies[f] >= lower)
        {
            freq_min_idx = f;
            break;
        }
    }
    for (int f = 0; f < freq_bins; f++)
    {
        if (frequencies[f] > upper)
        {
            freq_max_idx = f;
            break;
        }
    }

    // Determine time indices
    int time_min_idx = 0, time_max_idx = time_bins;
    for (int t = 0; t < time_bins; t++)
    {
        if (times[t] >= (midpoint - half_range))
        {
            time_min_idx = t;
            break;
        }
    }
    for (int t = 0; t < time_bins; t++)
    {
        if (times[t] > (midpoint + half_range))
        {
            time_max_idx = t;
            break;
        }
    }

    // Sum intensity
    for (int f = freq_min_idx; f < freq_max_idx; f++)
    {
        for (int t = time_min_idx; t < time_max_idx; t++)
        {
            double val = intensity[f * time_bins + t];
            if (!isnan(val))
                sum += val;
        }
    }

    return sum;
}
