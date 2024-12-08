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
void setup();
void loop();

bool butter_bandpass(float lowcut, float highcut, float *b, float *a);
void butter_bandpass_filter(const float *data, int n, const float *b, const float *a, float *output);
void compute_spectrogram(const float *signal, int signal_length, int fs,
                         float *frequencies, float *times, float *intensity,
                         int *freq_bins, int *time_bins);

float sum_intense(float lower, float upper, float half_range,
                  const float *frequencies, int freq_bins,
                  const float *times, int time_bins,
                  const float *intensity_dB_filtered, float midpoint);

float *find_midpoints(const float *data, int num_frames, int samplingFreq, int *num_midpoints);

static void normalize_and_filter_intensity(float *intensity, int freq_bins, int time_bins,
                                           float &min_intensity, float &max_intensity,
                                           float *intensity_dB_filtered);

// ------------------------------------------------------------
// Setup: Initialize and run analysis
// ------------------------------------------------------------
void setup()
{
    Serial.begin(9600);
    delay(2000);
    Serial.println("Starting Scrub Jay Detection...");

    // Setup filter coefficients
    butter_bandpass(LOWCUT_BP, HIGHCUT_BP, b_bp, a_bp);
    butter_bandpass(LOWCUT_MP, HIGHCUT_MP, b_mp, a_mp);

    // Filter signals for spectrogram and midpoint calculation
    butter_bandpass_filter(data, NUM_FRAMES, b_bp, a_bp, filtered_signal_bp);
    butter_bandpass_filter(data, NUM_FRAMES, b_mp, a_mp, filtered_signal_mp);

    // Compute spectrogram for BP filtered signal
    int freq_bins_bp = 0, time_bins_bp = 0;
    // We'll store frequencies, times, and intensity (freq_bins * time_bins)
    // Frequencies and times arrays:
    // Max freq_bins = NFFT/2+1 = 129 for WINDOW_SIZE=256
    // Max time_bins = (NUM_FRAMES - WINDOW_SIZE)/HOP_SIZE +1
    // For safety, assume max:
    float frequencies_bp[NFFT / 2 + 1];
    float times_bp[(NUM_FRAMES - WINDOW_SIZE) / HOP_SIZE + 1];
    float intensity_bp[(NFFT / 2 + 1) * ((NUM_FRAMES - WINDOW_SIZE) / HOP_SIZE + 1)];

    compute_spectrogram(filtered_signal_bp, NUM_FRAMES, SAMPLING_FREQ,
                        frequencies_bp, times_bp, intensity_bp, &freq_bins_bp, &time_bins_bp);

    // Normalize intensity and apply thresholds
    float min_intensity = FLT_MAX;
    float max_intensity = -FLT_MAX;
    float intensity_dB_filtered[(NFFT / 2 + 1) * ((NUM_FRAMES - WINDOW_SIZE) / HOP_SIZE + 1)];
    normalize_and_filter_intensity(intensity_bp, freq_bins_bp, time_bins_bp, min_intensity, max_intensity,
                                   intensity_dB_filtered);

    // Find midpoints from mp-filtered signal
    int num_midpoints = 0;
    float *midpoints = find_midpoints(data, NUM_FRAMES, SAMPLING_FREQ, &num_midpoints);
    if (!midpoints)
    {
        Serial.println("Failed to find midpoints.");
        return;
    }

    bool has_a_scrub = false;
    for (int idx = 0; idx < num_midpoints; idx++)
    {
        float midpoint = midpoints[idx];

        float sum_above = sum_intense(4500, 7500, 0.18, frequencies_bp, freq_bins_bp,
                                      times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
        float sum_middle = sum_intense(3500, 4000, 0.05, frequencies_bp, freq_bins_bp,
                                       times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
        float sum_below = sum_intense(500, 3000, 0.18, frequencies_bp, freq_bins_bp,
                                      times_bp, time_bins_bp, intensity_dB_filtered, midpoint);

        Serial.printlnf("Midpoint: %.3f s, Above: %.2f, Middle: %.2f, Below: %.2f",
                        midpoint, sum_above, sum_middle, sum_below);

        if (sum_middle < SUM_MIDDLE_THRESH && sum_above > SUM_ABOVE_THRESH && sum_below > SUM_BELOW_THRESH)
        {
            has_a_scrub = true;
            break;
        }
    }

    if (has_a_scrub)
    {
        Serial.println("This audio has a Scrub Jay! :)");
    }
    else
    {
        Serial.println("No Scrub Jay found. :(");
    }

    free(midpoints);
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
void compute_spectrogram(const float *signal, int signal_length, int fs,
                         float *frequencies, float *times, float *intensity,
                         int *freq_bins, int *time_bins)
{
    // One-sided FFT
    *freq_bins = NFFT / 2 + 1;
    *time_bins = (signal_length - WINDOW_SIZE) / HOP_SIZE + 1;

    // Compute frequency axis
    for (int i = 0; i < *freq_bins; i++)
    {
        frequencies[i] = (float)i * (float)fs / (float)NFFT;
    }

    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * HOP_SIZE;
        times[t] = (float)(start + WINDOW_SIZE / 2) / (float)fs;
    }

    // Temporary buffers for FFT
    float vReal[WINDOW_SIZE];
    float vImag[WINDOW_SIZE];

    // For each time frame
    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * HOP_SIZE;
        float sum = 0.0;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float val = (start + i < signal_length) ? signal[start + i] : 0.0;
            sum += val;
            vReal[i] = (float)val;
            vImag[i] = 0.0f;
        }
        float mean = sum / WINDOW_SIZE;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            vReal[i] -= (float)mean;
        }

        // Apply Hann window via PlainFFT
        fft.Windowing(vReal, WINDOW_SIZE, FFT_WIN_TYP_HANN, FFT_FORWARD);

        // Compute FFT
        fft.Compute(vReal, vImag, WINDOW_SIZE, FFT_FORWARD);
        fft.ComplexToMagnitude(vReal, vImag, WINDOW_SIZE);

        // Store intensity (power spectral density approx)
        // We won't do the exact scaling as previously, just store magnitude^2
        for (int f = 0; f < *freq_bins; f++)
        {
            float mag_squared = vReal[f] * vReal[f];
            intensity[f * (*time_bins) + t] = mag_squared;
        }
    }
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
    // Filter mp is already done: filtered_signal_mp
    // Compute spectrogram and find times > LOWER_THRESHOLD_DB
    int freq_bins_mp, time_bins_mp;
    float frequencies_mp[NFFT / 2 + 1];
    float times_mp[(num_frames - WINDOW_SIZE) / HOP_SIZE + 1];
    float intensity_mp[(NFFT / 2 + 1) * ((num_frames - WINDOW_SIZE) / HOP_SIZE + 1)];

    compute_spectrogram(filtered_signal_mp, num_frames, samplingFreq,
                        frequencies_mp, times_mp, intensity_mp,
                        &freq_bins_mp, &time_bins_mp);

    // Convert to dB and apply LOWER_THRESHOLD_DB
    for (int i = 0; i < freq_bins_mp; i++)
    {
        for (int j = 0; j < time_bins_mp; j++)
        {
            float val = intensity_mp[i * time_bins_mp + j];
            if (val > 0)
            {
                float dB = 10.0f * log10f(val / 1e-12f);
                if (dB < LOWER_THRESHOLD_DB)
                {
                    intensity_mp[i * time_bins_mp + j] = NAN;
                }
                else
                {
                    intensity_mp[i * time_bins_mp + j] = dB;
                }
            }
            else
            {
                intensity_mp[i * time_bins_mp + j] = NAN;
            }
        }
    }

    // Find time bins with any valid intensity
    bool valid_time_bins[time_bins_mp];
    for (int j = 0; j < time_bins_mp; j++)
    {
        valid_time_bins[j] = false;
        for (int i = 0; i < freq_bins_mp; i++)
        {
            if (!isnan(intensity_mp[i * time_bins_mp + j]))
            {
                valid_time_bins[j] = true;
                break;
            }
        }
    }

    // Extract blob times
    int num_blob_times = 0;
    for (int j = 0; j < time_bins_mp; j++)
    {
        if (valid_time_bins[j])
            num_blob_times++;
    }
    if (num_blob_times == 0)
    {
        *num_midpoints = 0;
        return NULL;
    }

    float *blob_times = (float *)malloc(num_blob_times * sizeof(float));
    int idx = 0;
    for (int j = 0; j < time_bins_mp; j++)
    {
        if (valid_time_bins[j])
        {
            blob_times[idx] = times_mp[j];
            idx++;
        }
    }

    // Cluster the blob_times
    float time_tolerance = 0.05;
    float min_blob_duration = 0.15;
    float *clusters[100]; // max 100 clusters
    int cluster_sizes[100];
    int num_clusters = 0;
    if (num_blob_times > 0)
    {
        clusters[0] = (float *)malloc(num_blob_times * sizeof(float));
        clusters[0][0] = blob_times[0];
        cluster_sizes[0] = 1;
        num_clusters = 1;
        for (int k = 1; k < num_blob_times; k++)
        {
            float dt = blob_times[k] - blob_times[k - 1];
            if (dt <= time_tolerance)
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

    free(blob_times);

    float *cluster_midpoints = (float *)malloc(num_clusters * sizeof(float));
    int count_mid = 0;
    for (int i = 0; i < num_clusters; i++)
    {
        float duration = clusters[i][cluster_sizes[i] - 1] - clusters[i][0];
        if (duration >= min_blob_duration)
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
        free(clusters[i]);
    }

    *num_midpoints = count_mid;
    if (count_mid == 0)
    {
        free(cluster_midpoints);
        return NULL;
    }

    return cluster_midpoints;
}
