//========================================================================
// classifier.c
//========================================================================
// A particle board compatible implementation of the Donut Classifier
#include "application.h"
#include "PlainFFT.h"
#include "Particle.h"
#include <math.h>
#include <float.h>
#include <stdbool.h>

// Function prototypes
bool butter_bandpass(float lowcut, float highcut, float *b, float *a);
void butter_bandpass_filter(float *data, int n, float *b, float *a, float *output);
void compute_spectrogram(float *signal, int signal_length, int fs, float **frequencies, float **times, float ***Sxx, int *freq_bins, int *time_bins);
float normalize_intensity(float value, float min, float max);
float sum_intense(float lower, float upper, float half_range, float *frequencies, int freq_bins, float *times, int time_bins, float **intensity_dB_filtered, float midpoint);
float *find_midpoints(float *data, int num_frames, int samplingFreq, int *num_midpoints);

void classify(float *data, int data_size)
{
    int num_frames = data_size; // sizeof(data) / sizeof(data[0]);

    int samplingFreq = 16000;

    float lowcut = 3000.0;
    float highcut = 7500.0;
    int order = 4;
    float b[9] = {0.0};
    float a[9] = {0.0};
    butter_bandpass(lowcut, highcut, b, a);

    // Apply Butterworth bandpass filter
    float *filtered_signal_bp = (float *)malloc(num_frames * sizeof(float));
    butter_bandpass_filter(data, num_frames, b, a, filtered_signal_bp);

    // Compute spectrogram of filtered_signal_bp
    float *frequencies_bp = NULL;
    float *times_bp = NULL;
    float **intensity_bp = NULL;
    int freq_bins_bp = 0, time_bins_bp = 0;

    compute_spectrogram(filtered_signal_bp, num_frames, samplingFreq, &frequencies_bp, &times_bp, &intensity_bp, &freq_bins_bp, &time_bins_bp);
    free(filtered_signal_bp);

    // Convert intensity to dB
    float min_intensity = DBL_MAX;
    float max_intensity = -DBL_MAX;
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
    for (int i = 0; i < freq_bins_bp; i++)
    {
        for (int j = 0; j < time_bins_bp; j++)
        {
            if (!isnan(intensity_bp[i][j]))
                intensity_bp[i][j] = (intensity_bp[i][j] - min_intensity) / (max_intensity - min_intensity);
            else
                intensity_bp[i][j] = NAN;
        }
    }

    float lower_threshold_dB_normalized = 0.70;
    float upper_threshold_dB_normalized = 0.85;

    // Apply normalized dB thresholds
    for (int i = 0; i < freq_bins_bp; i++)
    {
        for (int j = 0; j < time_bins_bp; j++)
        {
            if (intensity_bp[i][j] > lower_threshold_dB_normalized && intensity_bp[i][j] < upper_threshold_dB_normalized)
                intensity_bp[i][j] = intensity_bp[i][j];
            else
                intensity_bp[i][j] = NAN;
        }
    }

    // Scrub Jay Classify
    int num_midpoints = 0;
    float *midpoints = find_midpoints(data, num_frames, samplingFreq, &num_midpoints);
    // Serial.print("Number of Midpoints: ");
    // Serial.println(num_midpoints);

    if (midpoints == NULL)
    {
        Serial.println("Failed to find midpoints.");
        return;
    }
    bool has_a_scrub = false;
    for (int idx = 0; idx < num_midpoints; idx++)
    {
        float midpoint = midpoints[idx];

        float time_threshold = 0.18;

        float sum_above = sum_intense(5000, 7000, 0.18, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_bp, midpoint);
        float sum_middle = sum_intense(2500, 5000, 0.05, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_bp, midpoint);
        float sum_below = sum_intense(500, 2500, 0.18, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_bp, midpoint);

        Serial.print("Above intensities: ");
        Serial.println(sum_above);
        Serial.print("Middle intensities: ");
        Serial.println(sum_middle);
        Serial.print("Below intensities: ");
        Serial.println(sum_below);

        if (sum_middle < 100 && sum_above > 200 && sum_below > 150)
        {
            has_a_scrub = true;
            break;
        }
    }

    if (has_a_scrub)
    {
        Serial.println("We have a Scrub Jay! :)");
    }
    else
    {
        // Serial.println("We have no Scrub Jay! :(");
    }

    free(midpoints);
    for (int i = 0; i < freq_bins_bp; i++)
    {
        free(intensity_bp[i]);
    }
    free(intensity_bp);
    free(frequencies_bp);
    free(times_bp);
}

bool butter_bandpass(float lowcut, float highcut, float *b, float *a)
{
    if (lowcut == 1000 && highcut == 3000)
    {
        b[0] = 0.01020948;
        b[1] = 0.;
        b[2] = -0.04083792;
        b[3] = 0.;
        b[4] = 0.06125688;
        b[5] = 0.;
        b[6] = -0.04083792;
        b[7] = 0.;
        b[8] = 0.01020948;

        a[0] = 1.;
        a[1] = -4.56803686;
        a[2] = 9.95922498;
        a[3] = -13.49912589;
        a[4] = 12.43979269;
        a[5] = -7.94997696;
        a[6] = 3.43760562;
        a[7] = -0.92305481;
        a[8] = 0.1203896;
    }
    else if (lowcut == 3000 && highcut == 7500)
    {
        b[0] = 0.1362017;
        b[1] = 0.;
        b[2] = -0.5448068;
        b[3] = 0.;
        b[4] = 0.8172102;
        b[5] = 0.;
        b[6] = -0.5448068;
        b[7] = 0.;
        b[8] = 0.1362017;

        a[0] = 1.;
        a[1] = 2.60935592;
        a[2] = 2.32553038;
        a[3] = 1.20262614;
        a[4] = 1.11690211;
        a[5] = 0.76154474;
        a[6] = 0.10005124;
        a[7] = -0.0129829;
        a[8] = 0.02236815;
    }
    else
    {
        Serial.println("invalid bandpass range");
        return false;
    }
    return true;
}

void butter_bandpass_filter(float *data, int n, float *b, float *a, float *output)
{
    float w[9] = {0};

    for (int i = 0; i < n; i++)
    {
        float w0 = data[i];
        for (int j = 1; j < 9; j++)
        {
            w0 -= a[j] * w[j - 1];
        }

        float y = b[0] * w0;
        for (int j = 1; j < 9; j++)
        {
            y += b[j] * w[j - 1];
        }

        for (int j = 9 - 1; j > 0; j--)
        {
            w[j] = w[j - 1];
        }
        w[0] = w0;

        output[i] = y;
    }
}

void compute_spectrogram(float *signal, int signal_length, int fs,
                         float **frequencies, float **times, float ***Sxx,
                         int *freq_bins, int *time_bins)
{
    int window_size = 256;
    int noverlap = window_size / 8;
    int hop_size = window_size - noverlap;
    int nfft = window_size;
    float alpha = 0.25f;
    static float vReal[256];
    static float vImag[256];
    static float window[256];

    // Compute the number of frequency and time bins
    *freq_bins = nfft / 2 + 1;
    *time_bins = (signal_length - window_size) / hop_size + 1;

    // Allocate memory for frequencies, times, and the spectrogram matrix
    *frequencies = (float *)malloc((*freq_bins) * sizeof(float));
    *times = (float *)malloc((*time_bins) * sizeof(float));
    *Sxx = (float **)malloc((*freq_bins) * sizeof(float *));
    for (int i = 0; i < *freq_bins; i++)
    {
        (*Sxx)[i] = (float *)malloc((*time_bins) * sizeof(float));
    }

    // Compute frequency values
    for (int i = 0; i < *freq_bins; i++)
    {
        (*frequencies)[i] = (float)i * (float)fs / (float)nfft;
    }

    // Compute time values
    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;
        (*times)[t] = ((float)(start + window_size / 2)) / (float)fs;
    }

    // Create the Tukey window
    {
        float M = (float)(window_size + 1);
        if (alpha <= 0.0f)
        {
            // Rectangular window
            for (int i = 0; i < window_size; i++)
            {
                window[i] = 1.0f;
            }
        }
        else if (alpha >= 1.0f)
        {
            // Hann window
            for (int i = 0; i < window_size; i++)
            {
                window[i] = 0.5f * (1.0f - cosf(2.0f * (float)PI * (float)i / (M - 1.0f)));
            }
        }
        else
        {
            int width = (int)floorf(alpha * (M - 1.0f) / 2.0f);
            for (int n = 0; n <= width; n++)
            {
                window[n] = 0.5f * (1.0f + cosf((float)PI * (-1.0f + 2.0f * (float)n / (alpha * (M - 1.0f)))));
            }

            for (int n = width + 1; n <= (int)(M - width - 2); n++)
            {
                window[n] = 1.0f;
            }

            for (int n = (int)(M - width - 1); n < (int)M; n++)
            {
                window[n] = 0.5f * (1.0f + cosf((float)PI * (-2.0f / alpha + 1.0f + 2.0f * (float)n / (alpha * (M - 1.0f)))));
            }
        }
    }

    // Compute the window power (sum of squares)
    float U = 0.0f;
    for (int i = 0; i < window_size; i++)
    {
        U += window[i] * window[i];
    }
    U *= (float)fs;

    memset(vImag, 0, nfft * sizeof(float));
    PlainFFT fft;

    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;

        // Extract segment
        for (int i = 0; i < window_size; i++)
        {
            if (start + i < signal_length)
                vReal[i] = signal[start + i];
            else
                vReal[i] = 0.0f;
        }
        // Zero-padding if necessary
        for (int i = window_size; i < nfft; i++)
        {
            vReal[i] = 0.0f;
        }
        for (int i = 0; i < nfft; i++)
        {
            vImag[i] = 0.0f;
        }

        // Remove mean
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++)
        {
            sum += vReal[i];
        }
        float mean = sum / (float)window_size;
        for (int i = 0; i < window_size; i++)
        {
            vReal[i] -= mean;
        }

        // Apply Tukey window
        for (int i = 0; i < window_size; i++)
        {
            vReal[i] *= window[i];
        }

        // Compute FFT
        fft.Compute(vReal, vImag, (uint16_t)nfft, 0x01);

        // Compute Power Spectral Density
        for (int f = 0; f < *freq_bins; f++)
        {
            float real = vReal[f];
            float imag = vImag[f];
            float mag_squared = real * real + imag * imag;
            (*Sxx)[f][t] = mag_squared / U;
        }

        // Adjust for one-sided spectrum
        for (int f = 1; f < (*freq_bins - 1); f++)
        {
            (*Sxx)[f][t] *= 2.0f;
        }
    }
}

float sum_intense(float lower, float upper, float half_range, float *frequencies, int freq_bins, float *times, int time_bins, float **intensity_dB_filtered, float midpoint)
{
    // Find frequency indices
    int freq_min_idx = 0;
    while (freq_min_idx < freq_bins && frequencies[freq_min_idx] < lower)
        freq_min_idx++;

    int freq_max_idx = freq_bins - 1;
    while (freq_max_idx >= 0 && frequencies[freq_max_idx] > upper)
        freq_max_idx--;

    // Ensure indices are within bounds
    if (freq_min_idx >= freq_bins)
        freq_min_idx = freq_bins - 1;
    if (freq_max_idx < 0)
        freq_max_idx = 0;

    // Swap if needed
    if (freq_min_idx > freq_max_idx)
    {
        int temp = freq_min_idx;
        freq_min_idx = freq_max_idx;
        freq_max_idx = temp;
    }

    // Find time indices
    int time_min_idx = 0;
    while (time_min_idx < time_bins && times[time_min_idx] < midpoint - half_range)
        time_min_idx++;

    int time_max_idx = time_bins - 1;
    while (time_max_idx >= 0 && times[time_max_idx] > midpoint + half_range)
        time_max_idx--;

    // Ensure indices are within bounds
    if (time_min_idx >= time_bins)
        time_min_idx = time_bins - 1;
    if (time_max_idx < 0)
        time_max_idx = 0;

    // Swap if needed
    if (time_min_idx > time_max_idx)
    {
        int temp = time_min_idx;
        time_min_idx = time_max_idx;
        time_max_idx = temp;
    }

    float total_intensity = 0.0;

    for (int i = freq_min_idx; i <= freq_max_idx; i++)
    {
        for (int j = time_min_idx; j <= time_max_idx; j++)
        {
            if (!isnan(intensity_dB_filtered[i][j]))
            {
                total_intensity += intensity_dB_filtered[i][j];
            }
        }
    }
    return total_intensity;
}

float *find_midpoints(float *data, int num_frames, int samplingFreq, int *num_midpoints)
{

    float lower_threshold_dB = 45.0;

    float lowcut = 1000.0;
    float highcut = 3000.0;
    float b[9] = {0.0};
    float a[9] = {0.0};
    butter_bandpass(lowcut, highcut, b, a);

    // Apply Butterworth bandpass filter
    float *filtered_signal_mp = (float *)malloc(num_frames * sizeof(float));
    butter_bandpass_filter(data, num_frames, b, a, filtered_signal_mp);

    // Compute spectrogram of filtered_signal_bp
    float *frequencies_mp_bp = NULL;
    float *times_mp_bp = NULL;
    float **intensity_mp_bp = NULL;
    int freq_bins_mp_bp = 0, time_bins_mp_bp = 0;

    compute_spectrogram(filtered_signal_mp, num_frames, samplingFreq, &frequencies_mp_bp, &times_mp_bp, &intensity_mp_bp, &freq_bins_mp_bp, &time_bins_mp_bp);

    // Convert intensity to dB and normalize
    float min_intensity = DBL_MAX;
    float max_intensity = -DBL_MAX;
    for (int i = 0; i < freq_bins_mp_bp; i++)
    {
        for (int j = 0; j < time_bins_mp_bp; j++)
        {
            if (intensity_mp_bp[i][j] > 0)
            {
                intensity_mp_bp[i][j] = 10 * log10(intensity_mp_bp[i][j] / 1e-12);
                if (intensity_mp_bp[i][j] < min_intensity)
                    min_intensity = intensity_mp_bp[i][j];
                if (intensity_mp_bp[i][j] > max_intensity)
                    max_intensity = intensity_mp_bp[i][j];
            }
            else
            {
                intensity_mp_bp[i][j] = NAN;
            }
        }
    }

    // Apply lower_threshold_dB
    for (int i = 0; i < freq_bins_mp_bp; i++)
    {
        for (int j = 0; j < time_bins_mp_bp; j++)
        {
            if (intensity_mp_bp[i][j] > lower_threshold_dB)
                intensity_mp_bp[i][j] = intensity_mp_bp[i][j];
            else
                intensity_mp_bp[i][j] = NAN;
        }
    }
    // Collect times where there is any valid intensity
    bool *valid_time_bins = (bool *)malloc(time_bins_mp_bp * sizeof(bool));
    for (int j = 0; j < time_bins_mp_bp; j++)
    {
        valid_time_bins[j] = false;
        for (int i = 0; i < freq_bins_mp_bp; i++)
        {
            if (!isnan(intensity_mp_bp[i][j]))
            {
                valid_time_bins[j] = true;
                break;
            }
        }
    }

    // Collect the times where valid_time_bins[j] is true
    int num_blob_times = 0;
    for (int j = 0; j < time_bins_mp_bp; j++)
    {
        if (valid_time_bins[j])
            num_blob_times++;
    }
    float *blob_times = (float *)malloc(num_blob_times * sizeof(float));
    int idx = 0;
    for (int j = 0; j < time_bins_mp_bp; j++)
    {
        if (valid_time_bins[j])
        {
            blob_times[idx] = times_mp_bp[j];
            idx++;
        }
    }

    // Cluster the blob_times
    float time_tolerance = 0.05;
    float min_blob_duration = 0.15;

    int max_clusters = num_blob_times;
    float **clusters = (float **)malloc(max_clusters * sizeof(float *));
    int *cluster_sizes = (int *)malloc(max_clusters * sizeof(int));
    int num_clusters = 0;

    if (num_blob_times > 0)
    {
        clusters[0] = (float *)malloc(num_blob_times * sizeof(float));
        cluster_sizes[0] = 1;
        clusters[0][0] = blob_times[0];
        num_clusters = 1;

        for (int k = 1; k < num_blob_times; k++)
        {
            float delta_time = blob_times[k] - blob_times[k - 1];
            if (delta_time <= time_tolerance)
            {
                clusters[num_clusters - 1][cluster_sizes[num_clusters - 1]] = blob_times[k];
                cluster_sizes[num_clusters - 1]++;
            }
            else
            {
                clusters[num_clusters] = (float *)malloc((num_blob_times - k) * sizeof(float));
                clusters[num_clusters][0] = blob_times[k];
                cluster_sizes[num_clusters] = 1;
                num_clusters++;
            }
        }
    }

    // Calculate midpoints
    float *cluster_midpoints = (float *)malloc(num_clusters * sizeof(float));
    int count_midpoints = 0;

    for (int i = 0; i < num_clusters; i++)
    {
        float cluster_duration = clusters[i][cluster_sizes[i] - 1] - clusters[i][0];
        if (cluster_duration >= min_blob_duration)
        {
            float sum_times = 0.0;
            for (int j = 0; j < cluster_sizes[i]; j++)
            {
                sum_times += clusters[i][j];
            }
            float midpoint = sum_times / cluster_sizes[i];
            cluster_midpoints[count_midpoints] = midpoint;
            count_midpoints++;
        }
    }

    *num_midpoints = count_midpoints;

    // Free allocated memory
    for (int i = 0; i < num_clusters; i++)
    {
        free(clusters[i]);
    }
    free(clusters);
    free(cluster_sizes);
    free(blob_times);
    free(valid_time_bins);

    for (int i = 0; i < freq_bins_mp_bp; i++)
    {
        free(intensity_mp_bp[i]);
    }
    free(intensity_mp_bp);
    free(frequencies_mp_bp);
    free(times_mp_bp);
    free(filtered_signal_mp);

    return cluster_midpoints;
}
