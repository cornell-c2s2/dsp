#include "application.h" // For Particle devices
#include "PlainFFT.h"
#include "Particle.h"
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include "1809.h"

// ------------------------------------------------------------
// Configuration Parameters
// ------------------------------------------------------------
#define SAMPLING_FREQ 16000 // Example sample rate in Hz
#define WINDOW_SIZE 256     // FFT window size
#define NFFT WINDOW_SIZE
#define NOVERLAP (WINDOW_SIZE / 8)
#define HOP_SIZE (WINDOW_SIZE - NOVERLAP)
#define LOWER_THRESHOLD_DB 45.0 // For midpoint detection
#define FFT_FORWARD 0x01

// Bandpass filter ranges for midpoints and classification
#define LOWCUT_MP 1000.0
#define HIGHCUT_MP 3000.0
#define LOWCUT_BP 3000.0
#define HIGHCUT_BP 7500.0

// Thresholds for classification logic
#define LOWER_DB_NORM_THRESH 0.70
#define UPPER_DB_NORM_THRESH 0.85

// Classification Intensity Thresholds
#define SUM_ABOVE_THRESH 300.0
#define SUM_BELOW_THRESH 100.0
#define SUM_MIDDLE_THRESH 75.0

// ------------------------------------------------------------
// Hardcoded filter coefficients (from your original code)
// Adjust as needed. These are the "b" and "a" arrays for the chosen filters.
// ------------------------------------------------------------
static float b_bp[9], a_bp[9]; // For 3000-7500 Hz bandpass
static float b_mp[9], a_mp[9]; // For 1000-3000 Hz bandpass

// ------------------------------------------------------------
// Hardcoded Audio Data (Replace with your own)
// Ensure the array size matches NUM_FRAMES
// ------------------------------------------------------------
#define NUM_FRAMES 4803 // Example size, adjust as needed
// Insert your actual audio samples here (16-bit signed integers):

// ------------------------------------------------------------
// Global Arrays and Objects
// ------------------------------------------------------------

static float filtered_signal_bp[NUM_FRAMES];
static float filtered_signal_mp[NUM_FRAMES];

// PlainFFT object
PlainFFT fft;

// ------------------------------------------------------------
// Function Prototypes
// ------------------------------------------------------------

// Forward declarations
bool butter_bandpass(float lowcut, float highcut, float *b, float *a);
void butter_bandpass_filter(const float *data, int n, const float *b, const float *a, float *output);
float *find_midpoints(const float *data, int num_frames, int samplingFreq, int *num_midpoints);
float sum_intense(float lower, float upper, float half_range,
                  const float *frequencies, int freq_bins,
                  const float *times, int time_bins,
                  float **intensity_dB_filtered, float midpoint);

// New provided compute_spectrogram signature
void compute_spectrogram(float *signal, int signal_length, int fs,
                         float **frequencies, float **times, float ***Sxx,
                         int *freq_bins, int *time_bins);

// Utility functions (e.g., normalization and thresholding) adapted for float arrays
void normalize_and_threshold(float **Sxx, int freq_bins, int time_bins,
                             float *min_val, float *max_val,
                             float lower_norm_thresh, float upper_norm_thresh,
                             float **output);

// ------------------------------------------------------------
// Setup
// ------------------------------------------------------------
void setup()
{
    Serial.begin(9600);
    delay(2000);
    Serial.println("Setup start");

    // Initialize filters
    butter_bandpass(LOWCUT_BP, HIGHCUT_BP, b_bp, a_bp);
    butter_bandpass(LOWCUT_MP, HIGHCUT_MP, b_mp, a_mp);

    // Filter for BP
    static float filtered_signal_bp[NUM_FRAMES];
    butter_bandpass_filter(data, NUM_FRAMES, b_bp, a_bp, filtered_signal_bp);

    // Compute spectrogram using the new function
    float *frequencies_bp = NULL;
    float *times_bp = NULL;
    float **Sxx_bp = NULL;
    int freq_bins_bp = 0;
    int time_bins_bp = 0;

    // Check if time_bins would be positive:
    int tentative_time_bins = (NUM_FRAMES - WINDOW_SIZE) / HOP_SIZE + 1;
    if (tentative_time_bins <= 0)
    {
        Serial.println("Error: Not enough frames for given window/overlap settings.");
        return;
    }

    Serial.println("Before compute_spectrogram");
    compute_spectrogram(filtered_signal_bp, NUM_FRAMES, SAMPLING_FREQ,
                        &frequencies_bp, &times_bp, &Sxx_bp,
                        &freq_bins_bp, &time_bins_bp);

    Serial.printlnf("freq_bins: %d, time_bins: %d", freq_bins_bp, time_bins_bp);

    // Now Sxx_bp is allocated as a 2D array [freq_bins_bp][time_bins_bp]
    // Convert to dB, find min/max, normalize, and threshold
    float min_intensity = FLT_MAX;
    float max_intensity = -FLT_MAX;
    // Allocate output for normalized/thresholded data
    float **intensity_dB_filtered = (float **)malloc(freq_bins_bp * sizeof(float *));
    for (int i = 0; i < freq_bins_bp; i++)
    {
        intensity_dB_filtered[i] = (float *)malloc(time_bins_bp * sizeof(float));
    }

    normalize_and_threshold(Sxx_bp, freq_bins_bp, time_bins_bp,
                            &min_intensity, &max_intensity,
                            0.70f, 0.85f, // thresholds from your logic
                            intensity_dB_filtered);

    // Now find midpoints using MP-filter
    static float filtered_signal_mp[NUM_FRAMES];
    butter_bandpass_filter(data, NUM_FRAMES, b_mp, a_mp, filtered_signal_mp);

    int num_midpoints = 0;
    float *midpoints = find_midpoints(filtered_signal_mp, NUM_FRAMES, SAMPLING_FREQ, &num_midpoints);

    if (!midpoints)
    {
        Serial.println("No midpoints found.");
    }
    else
    {
        bool has_a_scrub = false;
        for (int i = 0; i < num_midpoints; i++)
        {
            float midpoint = midpoints[i];
            float sum_above = sum_intense(4500, 7500, 0.18, frequencies_bp, freq_bins_bp,
                                          times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
            float sum_middle = sum_intense(3500, 4000, 0.05, frequencies_bp, freq_bins_bp,
                                           times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
            float sum_below = sum_intense(500, 3000, 0.18, frequencies_bp, freq_bins_bp,
                                          times_bp, time_bins_bp, intensity_dB_filtered, midpoint);

            Serial.printlnf("Midpoint: %f Above: %f Middle: %f Below: %f",
                            midpoint, sum_above, sum_middle, sum_below);

            if (sum_middle < 75 && sum_above > 300 && sum_below > 100)
            {
                has_a_scrub = true;
                break;
            }
        }
        if (has_a_scrub)
        {
            Serial.println("Scrub Jay detected!");
        }
        else
        {
            Serial.println("No Scrub Jay detected.");
        }
        free(midpoints);
    }

    // Free memory allocated by compute_spectrogram and normalization
    for (int i = 0; i < freq_bins_bp; i++)
    {
        free(Sxx_bp[i]);
        free(intensity_dB_filtered[i]);
    }
    free(Sxx_bp);
    free(intensity_dB_filtered);
    free(frequencies_bp);
    free(times_bp);

    Serial.println("Done.");
}

void loop()
{
    // Nothing needed here if just analyzing once
}

// ------------------------------------------------------------
// Filter design and application
// ------------------------------------------------------------
bool butter_bandpass(float lowcut, float highcut, float *b, float *a)
{
    // Hard-coded coefficients from your original code:
    // For (3000, 7500)
    if (fabs(lowcut - 3000.0) < 1e-6 && fabs(highcut - 7500.0) < 1e-6)
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
    else if (fabs(lowcut - 1000.0) < 1e-6 && fabs(highcut - 3000.0) < 1e-6)
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
    else
    {
        Serial.println("Invalid bandpass range chosen!");
        return false;
    }
    return true;
}

void butter_bandpass_filter(const float *data, int n, const float *b, const float *a, float *output)
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
        for (int j = 8; j > 0; j--)
        {
            w[j] = w[j - 1];
        }
        w[0] = w0;
        output[i] = y;
    }
}

// ------------------------------------------------------------
// Spectrogram computation using PlainFFT
// ------------------------------------------------------------
void compute_spectrogram(float *signal, int signal_length, int fs,
                         float **frequencies, float **times, float ***Sxx,
                         int *freq_bins, int *time_bins)
{
    int window_size = 256;
    int noverlap = window_size / 8;        // 32 points overlap
    int hop_size = window_size - noverlap; // 224 points step size
    int nfft = window_size;
    float alpha = 0.25f; // Tukey window parameter

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
    float *window = (float *)malloc(window_size * sizeof(float));
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

    // Allocate arrays for FFT
    float *vReal = (float *)malloc(nfft * sizeof(float));
    float *vImag = (float *)malloc(nfft * sizeof(float));
    memset(vImag, 0, nfft * sizeof(float));

    // Create PlainFFT instance
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
        fft.Compute(vReal, vImag, (uint16_t)nfft, FFT_FORWARD);

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

    // Clean up
    free(vReal);
    free(vImag);
    free(window);
}

// ------------------------------------------------------------
// Normalize intensity, convert to dB, and filter by thresholds
// ------------------------------------------------------------
static void normalize_and_filter_intensity(float *intensity, int freq_bins, int time_bins,
                                           float &min_intensity, float &max_intensity,
                                           float *intensity_dB_filtered)
{
    // Convert to dB and find min/max
    for (int i = 0; i < freq_bins; i++)
    {
        for (int j = 0; j < time_bins; j++)
        {
            float val = intensity[i * time_bins + j];
            if (val > 0)
            {
                float dB = 10.0f * log10f(val / 1e-12f);
                if (dB < min_intensity)
                    min_intensity = dB;
                if (dB > max_intensity)
                    max_intensity = dB;
                intensity[i * time_bins + j] = dB;
            }
            else
            {
                intensity[i * time_bins + j] = NAN;
            }
        }
    }

    // Normalize and apply thresholds
    for (int i = 0; i < freq_bins; i++)
    {
        for (int j = 0; j < time_bins; j++)
        {
            float dB = intensity[i * time_bins + j];
            if (!isnan(dB))
            {
                float norm = (dB - min_intensity) / (max_intensity - min_intensity);
                if (norm > LOWER_DB_NORM_THRESH && norm < UPPER_DB_NORM_THRESH)
                {
                    intensity_dB_filtered[i * time_bins + j] = norm;
                }
                else
                {
                    intensity_dB_filtered[i * time_bins + j] = NAN;
                }
            }
            else
            {
                intensity_dB_filtered[i * time_bins + j] = NAN;
            }
        }
    }
}

// ------------------------------------------------------------
// Summation of intensity in freq/time ranges around a midpoint
// ------------------------------------------------------------
float sum_intense(float lower, float upper, float half_range,
                  const float *frequencies, int freq_bins,
                  const float *times, int time_bins,
                  const float *intensity_dB_filtered, float midpoint)
{
    int freq_min_idx = 0;
    while (freq_min_idx < freq_bins && frequencies[freq_min_idx] < (float)lower)
        freq_min_idx++;

    int freq_max_idx = freq_bins - 1;
    while (freq_max_idx >= 0 && frequencies[freq_max_idx] > (float)upper)
        freq_max_idx--;

    if (freq_min_idx > freq_max_idx)
    {
        int temp = freq_min_idx;
        freq_min_idx = freq_max_idx;
        freq_max_idx = temp;
    }

    int time_min_idx = 0;
    while (time_min_idx < time_bins && times[time_min_idx] < (float)(midpoint - half_range))
        time_min_idx++;

    int time_max_idx = time_bins - 1;
    while (time_max_idx >= 0 && times[time_max_idx] > (float)(midpoint + half_range))
        time_max_idx--;

    if (time_min_idx > time_max_idx)
    {
        int temp = time_min_idx;
        time_min_idx = time_max_idx;
        time_max_idx = temp;
    }

    float total_intensity = 0.0;
    for (int i = freq_min_idx; i <= freq_max_idx; i++)
    {
        if (i < 0 || i >= freq_bins)
            continue;
        for (int j = time_min_idx; j <= time_max_idx; j++)
        {
            if (j < 0 || j >= time_bins)
                continue;
            float val = intensity_dB_filtered[i * time_bins + j];
            if (!isnan(val))
                total_intensity += val;
        }
    }
    return total_intensity;
}

// ------------------------------------------------------------
// Find midpoints (clusters of activity) in the mp-filtered signal
// ------------------------------------------------------------
float *find_midpoints(const float *data, int num_frames, int samplingFreq, int *num_midpoints)
{
    *num_midpoints = 0;

    // Filter the signal in MP range
    static float b_mp[9], a_mp[9];
    if (!butter_bandpass(MP_LOWCUT, MP_HIGHCUT, b_mp, a_mp))
    {
        Serial.println("Failed to set MP bandpass filter.");
        return NULL;
    }

    float *filtered_signal_mp = (float *)malloc(num_frames * sizeof(float));
    if (!filtered_signal_mp)
    {
        Serial.println("Memory allocation failed in find_midpoints().");
        return NULL;
    }

    butter_bandpass_filter(data, num_frames, b_mp, a_mp, filtered_signal_mp);

    // Compute spectrogram
    float *frequencies_mp = NULL;
    float *times_mp = NULL;
    float **Sxx_mp = NULL;
    int freq_bins_mp = 0;
    int time_bins_mp = 0;

    compute_spectrogram(filtered_signal_mp, num_frames, samplingFreq,
                        &frequencies_mp, &times_mp, &Sxx_mp,
                        &freq_bins_mp, &time_bins_mp);

    free(filtered_signal_mp);

    if (time_bins_mp <= 0 || freq_bins_mp <= 0)
    {
        Serial.println("Invalid time_bins or freq_bins in find_midpoints()");
        // Free allocated memory
        if (frequencies_mp)
            free(frequencies_mp);
        if (times_mp)
            free(times_mp);
        if (Sxx_mp)
        {
            for (int i = 0; i < freq_bins_mp; i++)
            {
                free(Sxx_mp[i]);
            }
            free(Sxx_mp);
        }
        return NULL;
    }

    // Convert to dB and apply threshold
    for (int i = 0; i < freq_bins_mp; i++)
    {
        for (int j = 0; j < time_bins_mp; j++)
        {
            float val = Sxx_mp[i][j];
            if (val > 0.0f)
            {
                float dB = 10.0f * log10f(val / 1e-12f);
                if (dB < LOWER_THRESHOLD_DB)
                {
                    Sxx_mp[i][j] = NAN; // Below threshold
                }
                else
                {
                    Sxx_mp[i][j] = dB; // Keep in dB
                }
            }
            else
            {
                Sxx_mp[i][j] = NAN;
            }
        }
    }

    // Identify time bins with any valid intensity
    bool *valid_time_bins = (bool *)malloc(time_bins_mp * sizeof(bool));
    if (!valid_time_bins)
    {
        Serial.println("Memory allocation failed for valid_time_bins.");
        // Free allocated memory
        for (int i = 0; i < freq_bins_mp; i++)
            free(Sxx_mp[i]);
        free(Sxx_mp);
        free(frequencies_mp);
        free(times_mp);
        return NULL;
    }

    for (int j = 0; j < time_bins_mp; j++)
    {
        valid_time_bins[j] = false;
        for (int i = 0; i < freq_bins_mp; i++)
        {
            if (!isnan(Sxx_mp[i][j]))
            {
                valid_time_bins[j] = true;
                break;
            }
        }
    }

    // Collect blob times
    int num_blob_times = 0;
    for (int j = 0; j < time_bins_mp; j++)
    {
        if (valid_time_bins[j])
            num_blob_times++;
    }

    float *blob_times = NULL;
    if (num_blob_times > 0)
    {
        blob_times = (float *)malloc(num_blob_times * sizeof(float));
        if (!blob_times)
        {
            Serial.println("Memory allocation failed for blob_times.");
            free(valid_time_bins);
            for (int i = 0; i < freq_bins_mp; i++)
                free(Sxx_mp[i]);
            free(Sxx_mp);
            free(frequencies_mp);
            free(times_mp);
            return NULL;
        }

        int idx = 0;
        for (int j = 0; j < time_bins_mp; j++)
        {
            if (valid_time_bins[j])
            {
                blob_times[idx] = (float)times_mp[j];
                idx++;
            }
        }
    }

    free(valid_time_bins);

    // No blobs found
    if (num_blob_times == 0)
    {
        for (int i = 0; i < freq_bins_mp; i++)
            free(Sxx_mp[i]);
        free(Sxx_mp);
        free(frequencies_mp);
        free(times_mp);
        return NULL;
    }

    // Cluster the blob_times
    // Upper bound on clusters is num_blob_times (worst case: all separate)
    float **clusters = (float **)malloc(num_blob_times * sizeof(float *));
    int *cluster_sizes = (int *)malloc(num_blob_times * sizeof(int));
    if (!clusters || !cluster_sizes)
    {
        Serial.println("Memory allocation failed for clustering.");
        free(blob_times);
        if (clusters)
            free(clusters);
        if (cluster_sizes)
            free(cluster_sizes);
        for (int i = 0; i < freq_bins_mp; i++)
            free(Sxx_mp[i]);
        free(Sxx_mp);
        free(frequencies_mp);
        free(times_mp);
        return NULL;
    }

    int num_clusters = 0;
    clusters[0] = (float *)malloc(num_blob_times * sizeof(float));
    if (!clusters[0])
    {
        Serial.println("Memory allocation failed for first cluster.");
        free(blob_times);
        free(clusters);
        free(cluster_sizes);
        for (int i = 0; i < freq_bins_mp; i++)
            free(Sxx_mp[i]);
        free(Sxx_mp);
        free(frequencies_mp);
        free(times_mp);
        return NULL;
    }
    clusters[0][0] = blob_times[0];
    cluster_sizes[0] = 1;
    num_clusters = 1;

    for (int k = 1; k < num_blob_times; k++)
    {
        float dt = blob_times[k] - blob_times[k - 1];
        if (dt <= TIME_TOLERANCE)
        {
            // Same cluster
            clusters[num_clusters - 1][cluster_sizes[num_clusters - 1]] = blob_times[k];
            cluster_sizes[num_clusters - 1]++;
        }
        else
        {
            // New cluster
            clusters[num_clusters] = (float *)malloc((num_blob_times - k) * sizeof(float));
            if (!clusters[num_clusters])
            {
                Serial.println("Memory allocation failed for a new cluster.");
                // Free all allocated clusters and arrays
                for (int c = 0; c < num_clusters; c++)
                    free(clusters[c]);
                free(blob_times);
                free(clusters);
                free(cluster_sizes);
                for (int i = 0; i < freq_bins_mp; i++)
                    free(Sxx_mp[i]);
                free(Sxx_mp);
                free(frequencies_mp);
                free(times_mp);
                return NULL;
            }
            clusters[num_clusters][0] = blob_times[k];
            cluster_sizes[num_clusters] = 1;
            num_clusters++;
        }
    }

    free(blob_times);

    // Calculate midpoints of clusters that last >= MIN_BLOB_DURATION
    float *cluster_midpoints = (float *)malloc(num_clusters * sizeof(float));
    if (!cluster_midpoints)
    {
        Serial.println("Memory allocation failed for cluster_midpoints.");
        // Free clusters
        for (int c = 0; c < num_clusters; c++)
            free(clusters[c]);
        free(clusters);
        free(cluster_sizes);
        for (int i = 0; i < freq_bins_mp; i++)
            free(Sxx_mp[i]);
        free(Sxx_mp);
        free(frequencies_mp);
        free(times_mp);
        return NULL;
    }

    int count_mid = 0;
    for (int i = 0; i < num_clusters; i++)
    {
        float duration = clusters[i][cluster_sizes[i] - 1] - clusters[i][0];
        if (duration >= MIN_BLOB_DURATION)
        {
            float sum_t = 0.0;
            for (int j = 0; j < cluster_sizes[i]; j++)
            {
                sum_t += clusters[i][j];
            }
            float midpoint = sum_t / cluster_sizes[i];
            cluster_midpoints[count_mid] = midpoint;
            count_mid++;
        }
    }

    // Free clusters
    for (int c = 0; c < num_clusters; c++)
        free(clusters[c]);
    free(clusters);
    free(cluster_sizes);

    // Free the Sxx arrays and frequency/time arrays
    for (int i = 0; i < freq_bins_mp; i++)
        free(Sxx_mp[i]);
    free(Sxx_mp);
    free(frequencies_mp);
    free(times_mp);

    if (count_mid == 0)
    {
        free(cluster_midpoints);
        *num_midpoints = 0;
        return NULL;
    }

    *num_midpoints = count_mid;
    return cluster_midpoints;
}
