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
#define PI 3.141592653589793

// Function prototypes
int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels);
bool butter_bandpass(double lowcut, double highcut, double *b, double *a);
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output);
void compute_spectrogram(double *signal, int signal_length, int fs, double **frequencies, double **times, double ***intensity, int *freq_bins, int *time_bins);
double normalize_intensity(double value, double min, double max);
double sum_intense(double lower, double upper, double half_range, double *frequencies, int freq_bins, double *times, int time_bins, double **intensity_dB_filtered, double midpoint);
double *find_midpoints(double *data, int num_frames, int samplingFreq, int *num_midpoints);

int main()
{

    char folder[] = "16k";

    struct dirent *entry;
    DIR *directory = opendir(folder);
    while ((entry = readdir(directory)) != NULL)
    {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        {

            char audioFile[MAX_FILENAME];
            snprintf(audioFile, sizeof(audioFile), "%s/%s", folder, entry->d_name);

            bool showGraphsAndPrint = true;

            // Step 1: Read the WAV file
            // const char *filename = "testing/2287-sj.wav";
            int16_t *wav_data = NULL;
            int samplingFreq, num_frames, num_channels;

            read_wav_file(audioFile, &wav_data, &samplingFreq, &num_frames, &num_channels);

            printf("Loaded WAV file: %s\n", audioFile);
            printf("Sample Rate: %d Hz\n", samplingFreq);
            printf("Channels: %d (output as mono)\n", num_channels);
            printf("Total Samples: %d\n", num_frames);

            // Step 2: Normalize and convert WAV data to double
            double *data = malloc(num_frames * sizeof(double));
            for (int i = 0; i < num_frames; i++)
            {
                data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
            }

            free(wav_data); // No longer needed

            // Save Data to files
            char dataName[MAX_FILENAME];
            snprintf(dataName, sizeof(dataName), "data/%s.txt", entry->d_name);
            FILE *_data = fopen(dataName, "w");
            for (int t = 0; t < num_frames; t++)
            {
                if (t != num_frames - 1)
                {

                    fprintf(_data, "%f,", (data[t]));
                }
                else
                {
                    fprintf(_data, "%f", (data[t]));
                }
            }
            fclose(_data);
            printf("_data saved to '%s'\n", dataName);

            // Step 3: Apply Butterworth bandpass filter
            double *filtered = malloc(num_frames * sizeof(double));
            double lowcut = 3000.0;
            double highcut = 7500.0;
            int order = 4;
            double b[9] = {0.0};
            double a[9] = {0.0};
            butter_bandpass(lowcut, highcut, b, a);

            // Calculate Spectrogram with new bandpass

            // Apply Butterworth bandpass filter
            double *filtered_signal_bp = (double *)malloc(num_frames * sizeof(double));
            butter_bandpass_filter(data, num_frames, b, a, filtered_signal_bp);

            // Compute spectrogram of filtered_signal_bp
            double *frequencies_bp = NULL;
            double *times_bp = NULL;
            double **intensity_bp = NULL;
            int freq_bins_bp = 0, time_bins_bp = 0;

            // this is right now
            compute_spectrogram(filtered_signal_bp, num_frames, samplingFreq, &frequencies_bp, &times_bp, &intensity_bp, &freq_bins_bp, &time_bins_bp);

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
            printf("%f", min_intensity);
            printf("%f", max_intensity);
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

            double lower_threshold_dB_normalized = 0.70;
            double upper_threshold_dB_normalized = 0.85;

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

            // Scrub Jay Classify
            int num_midpoints = 0;
            double *midpoints = find_midpoints(data, num_frames, samplingFreq, &num_midpoints);

            if (midpoints == NULL)
            {
                printf("Failed to find midpoints.\n");
                // Handle error
                return 1;
            }
            printf("%d\n", num_midpoints);
            bool has_a_scrub = false;
            for (int idx = 0; idx < num_midpoints; idx++)
            {
                double midpoint = midpoints[idx];

                double time_threshold = 0.18;

                double sum_above = sum_intense(4500, 7500, 0.18, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
                double sum_middle = sum_intense(3500, 4000, 0.05, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_dB_filtered, midpoint);
                double sum_below = sum_intense(500, 3000, 0.18, frequencies_bp, freq_bins_bp, times_bp, time_bins_bp, intensity_dB_filtered, midpoint);

                if (showGraphsAndPrint)
                {
                    printf("Above: %f\n", sum_above);
                    printf("Middle: %f\n", sum_middle);
                    printf("Below: %f\n\n", sum_below);
                }

                if (sum_middle < 75 && sum_above > 300 && sum_below > 100)
                {
                    has_a_scrub = true;
                    break; // We can stop after finding a Scrub Jay
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
            free(data);
            free(midpoints);

            for (int i = 0; i < freq_bins_bp; i++)
            {
                free(intensity_normalized[i]);
                free(intensity_dB_filtered[i]);
            }
            free(intensity_normalized);
            free(intensity_dB_filtered);
        }
    }
    return 0;
}

int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Error opening file");
        return 0;
    }

    char chunk_id[4];
    uint32_t chunk_size;
    uint16_t audio_format, channels;
    uint32_t sample_rate_local, byte_rate;
    uint16_t block_align, bits_per_sample;
    uint32_t data_chunk_size;

    // Read RIFF header
    fread(chunk_id, 1, 4, file); // "RIFF"
    if (strncmp(chunk_id, "RIFF", (size_t)4) != 0)
    {
        fprintf(stderr, "Not a valid RIFF file.\n");
        fclose(file);
        return 0;
    }
    fread(&chunk_size, 4, 1, file); // Chunk size
    fread(chunk_id, 1, 4, file);    // "WAVE"
    if (strncmp(chunk_id, "WAVE", (size_t)4) != 0)
    {
        fprintf(stderr, "Not a valid WAVE file.\n");
        fclose(file);
        return 0;
    }

    // Read fmt subchunk
    fread(chunk_id, 1, 4, file); // "fmt "
    if (strncmp(chunk_id, "fmt ", (size_t)4) != 0)
    {
        fprintf(stderr, "Missing 'fmt ' subchunk.\n");
        fclose(file);
        return 0;
    }
    fread(&chunk_size, 4, 1, file);        // Subchunk1Size
    fread(&audio_format, 2, 1, file);      // AudioFormat
    fread(&channels, 2, 1, file);          // NumChannels
    fread(&sample_rate_local, 4, 1, file); // SampleRate
    fread(&byte_rate, 4, 1, file);         // ByteRate
    fread(&block_align, 2, 1, file);       // BlockAlign
    fread(&bits_per_sample, 2, 1, file);   // BitsPerSample

    if (audio_format != 1)
    {
        fprintf(stderr, "Unsupported audio format: %d\n", audio_format);
        fclose(file);
        return 0;
    }

    // Read data subchunk
    fread(chunk_id, 1, 4, file); // "data"
    while (strncmp(chunk_id, "data", (size_t)4) != 0)
    {
        fread(&chunk_size, 4, 1, file);
        fseek(file, chunk_size, SEEK_CUR);
        fread(chunk_id, 1, 4, file);
    }
    fread(&data_chunk_size, 4, 1, file);

    // Calculate the number of samples per channel
    int total_samples = data_chunk_size / (bits_per_sample / 8);
    int samples_per_channel = total_samples / channels;

    // Allocate memory for the first channel's data
    *data = (int16_t *)malloc(samples_per_channel * sizeof(int16_t));
    if (!*data)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        return 0;
    }

    // Read audio data, only taking the first channel
    for (int i = 0; i < samples_per_channel; i++)
    {
        fread(&((*data)[i]), sizeof(int16_t), 1, file);          // Read the first channel
        fseek(file, (channels - 1) * sizeof(int16_t), SEEK_CUR); // Skip remaining channels
    }

    // Set metadata
    *sample_rate = sample_rate_local;
    *num_samples = samples_per_channel;
    *num_channels = 1; // Output is now single-channel

    fclose(file);
    return 1;
}

/**
 * Designs a Butterworth bandpass filter using the bilinear transform method.
 * Supports second-order (order=2) filters.
 *
 * @param lowcut Low cut-off frequency in Hz
 * @param highcut High cut-off frequency in Hz
 * @param fs Sampling frequency in Hz
 * @param order Filter order (only supports order=4)
 * @param b Output array for numerator coefficients (size 9)
 * @param a Output array for denominator coefficients (size 9)
 * @return true on success, false on failure
 */
bool butter_bandpass(double lowcut, double highcut, double *b, double *a)
{
    if (lowcut == 1000 && highcut == 3000)
    {
        // b[0] = 0.00021314;
        // b[1] = 0.;
        // b[2] = -0.00085255;
        // b[3] = 0.;
        // b[4] = 0.00127883;
        // b[5] = 0.;
        // b[6] = -0.00085255;
        // b[7] = 0.;
        // b[8] = 0.00021314;

        // a[0] = 1.;
        // a[1] = -7.12847885;
        // a[2] = 22.41882266;
        // a[3] = -40.62891245;
        // a[4] = 46.40780141;
        // a[5] = -34.21333503;
        // a[6] = 15.89913237;
        // a[7] = -4.25840048;
        // a[8] = 0.50337536;  b[0] = 0.00021314;
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
        // b[0] = 0.00386952;
        // b[1] = 0.;
        // b[2] = -0.01547807;
        // b[3] = 0.;
        // b[4] = 0.02321711;
        // b[5] = 0.;
        // b[6] = -0.01547807;
        // b[7] = 0.;
        // b[8] = 0.00386952;

        // a[0] = 1.;
        // a[1] = -5.22664543;
        // a[2] = 12.83819436;
        // a[3] = -19.22549589;
        // a[4] = 19.15517565;
        // a[5] = -12.98213646;
        // a[6] = 5.84957071;
        // a[7] = -1.60753218;
        // a[8] = 0.2088483;
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
        fprintf(stderr, "invalid bandpass range\n");
        return false;
    }
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
    double w[9] = {0}; // State variables initialized to zero

    for (int i = 0; i < n; i++)
    {
        double w0 = data[i];
        for (int j = 1; j < 9; j++)
        {
            w0 -= a[j] * w[j - 1];
        }

        double y = b[0] * w0;
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

void compute_spectrogram(double *signal, int signal_length, int fs,
                         double **frequencies, double **times, double ***Sxx,
                         int *freq_bins, int *time_bins)
{
    int window_size = 64;
    int noverlap = window_size / 8;        // 32 points overlap
    int hop_size = window_size - noverlap; // 224 points step size
    int nfft = window_size;
    double alpha = 0.25; // Tukey window parameter

    // Compute the number of frequency bins and time bins
    *freq_bins = nfft / 2 + 1; // One-sided spectrum
    *time_bins = (signal_length - window_size) / hop_size + 1;

    // Allocate memory for frequencies, times, and the spectrogram matrix
    *frequencies = (double *)malloc((*freq_bins) * sizeof(double));
    *times = (double *)malloc((*time_bins) * sizeof(double));
    *Sxx = (double **)malloc((*freq_bins) * sizeof(double *));
    for (int i = 0; i < *freq_bins; i++)
    {
        (*Sxx)[i] = (double *)malloc((*time_bins) * sizeof(double));
    }

    // Compute frequency values
    for (int i = 0; i < *freq_bins; i++)
    {
        (*frequencies)[i] = (double)i * fs / nfft;
    }

    // Compute time values
    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;
        (*times)[t] = ((double)(start + window_size / 2)) / fs;
    }

    // Create the Tukey window
    double *window = (double *)malloc(window_size * sizeof(double));
    double N_minus_1 = (double)(window_size - 1);
    int M = window_size + 1; // Total number of points + 1 bc symmetric idk the python did this

    if (alpha <= 0)
    {
        // Rectangular window
        for (int i = 0; i < window_size; i++)
        {
            window[i] = 1.0;
        }
    }
    else if (alpha >= 1.0)
    {
        // Hann window
        for (int i = 0; i < window_size; i++)
        {
            window[i] = 0.5 * (1 - cos(2 * PI * i / (M - 1)));
        }
    }
    else
    {
        int width = (int)floor(alpha * (M - 1) / 2.0);
        for (int n = 0; n <= width; n++)
        {
            window[n] = 0.5 * (1 + cos(PI * (-1 + 2.0 * n / (alpha * (M - 1)))));
        }

        for (int n = width + 1; n <= M - width - 2; n++)
        {
            window[n] = 1.0;
        }

        for (int n = M - width - 1; n < M; n++)
        {
            window[n] = 0.5 * (1 + cos(PI * (-2.0 / alpha + 1 + 2.0 * n / (alpha * (M - 1)))));
        }
    }

    // Compute the window power (sum of squares)
    // this U is equal to 1/scale in the python implementation
    double U = 0.0;
    for (int i = 0; i < window_size; i++)
    {
        U += window[i] * window[i];
    }
    U *= fs; // Include sampling frequency in scaling

    // Allocate memory for FFT input and output
    double *segment = (double *)malloc(nfft * sizeof(double));
    fftw_complex *out = (fftw_complex *)fftw_malloc((*freq_bins) * sizeof(fftw_complex));

    // Create FFTW plan
    fftw_plan p = fftw_plan_dft_r2c_1d(nfft, segment, out, FFTW_ESTIMATE);

    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;

        // Extract the segment and apply zero-padding if necessary
        for (int i = 0; i < window_size; i++)
        {
            if (start + i < signal_length)
                segment[i] = signal[start + i];
            else
                segment[i] = 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < window_size; i++)
        {
            sum += segment[i];
        }

        double mean = sum / window_size;
        for (int i = 0; i < window_size; i++)
        {
            segment[i] -= mean;
        }

        // Apply the window to the segment
        for (int i = 0; i < window_size; i++)
        {
            segment[i] *= window[i];
        }

        // Execute the FFT
        fftw_execute(p);

        // Compute the power spectral density and apply scaling
        for (int f = 0; f < *freq_bins; f++)
        {
            double real = out[f][0];
            double imag = out[f][1];
            double mag_squared = real * real + imag * imag;
            (*Sxx)[f][t] = mag_squared / U;
        }

        // Adjust scaling for one-sided spectrum
        for (int f = 1; f < *freq_bins - 1; f++)
        {
            (*Sxx)[f][t] *= 2.0;
        }
    }

    // Clean up
    fftw_destroy_plan(p);
    fftw_free(out);
    free(segment);
    free(window);
}

double sum_intense(double lower, double upper, double half_range, double *frequencies, int freq_bins, double *times, int time_bins, double **intensity_dB_filtered, double midpoint)
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

    double total_intensity = 0.0;

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

double *find_midpoints(double *data, int num_frames, int samplingFreq, int *num_midpoints)
{

    double lower_threshold_dB = 45.0;

    double lowcut = 1000.0;
    double highcut = 3000.0;
    double b[9] = {0.0};
    double a[9] = {0.0};
    butter_bandpass(lowcut, highcut, b, a);

    // Apply Butterworth bandpass filter
    double *filtered_signal_mp = (double *)malloc(num_frames * sizeof(double));
    butter_bandpass_filter(data, num_frames, b, a, filtered_signal_mp);

    // Compute spectrogram of filtered_signal_bp
    double *frequencies_mp_bp = NULL;
    double *times_mp_bp = NULL;
    double **intensity_mp_bp = NULL;
    int freq_bins_mp_bp = 0, time_bins_mp_bp = 0;

    // this is right now
    compute_spectrogram(filtered_signal_mp, num_frames, samplingFreq, &frequencies_mp_bp, &times_mp_bp, &intensity_mp_bp, &freq_bins_mp_bp, &time_bins_mp_bp);

    // Convert intensity to dB and normalize
    double min_intensity = DBL_MAX;
    double max_intensity = -DBL_MAX;
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
    double **intensity_dB_filtered = (double **)malloc(freq_bins_mp_bp * sizeof(double *));
    for (int i = 0; i < freq_bins_mp_bp; i++)
    {
        intensity_dB_filtered[i] = (double *)malloc(time_bins_mp_bp * sizeof(double));
        for (int j = 0; j < time_bins_mp_bp; j++)
        {
            if (intensity_mp_bp[i][j] > lower_threshold_dB)
                intensity_dB_filtered[i][j] = intensity_mp_bp[i][j];
            else
                intensity_dB_filtered[i][j] = NAN;
        }
    }
    // Collect times where there is any valid intensity
    bool *valid_time_bins = (bool *)malloc(time_bins_mp_bp * sizeof(bool));
    for (int j = 0; j < time_bins_mp_bp; j++)
    {
        valid_time_bins[j] = false;
        for (int i = 0; i < freq_bins_mp_bp; i++)
        {
            if (!isnan(intensity_dB_filtered[i][j]))
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
    printf("%d", num_blob_times);
    double *blob_times = (double *)malloc(num_blob_times * sizeof(double));
    int idx = 0;
    for (int j = 0; j < time_bins_mp_bp; j++)
    {
        if (valid_time_bins[j])
        {
            blob_times[idx] = times_mp_bp[j];
            idx++;
        }
    }

    // FILE *_blobtimes = fopen("_blobtimes.txt", "w");
    // for (int t = 0; t < num_blob_times; t++)
    // {
    //     fprintf(_blobtimes, " %f", (blob_times[t]));
    // }
    // fclose(_blobtimes);
    // printf("_blobtimes data saved to '_blobtimes.txt'\n");

    // Cluster the blob_times
    double time_tolerance = 0.05;    // seconds
    double min_blob_duration = 0.15; // seconds

    int max_clusters = num_blob_times;
    double **clusters = (double **)malloc(max_clusters * sizeof(double *));
    int *cluster_sizes = (int *)malloc(max_clusters * sizeof(int));
    int num_clusters = 0;

    if (num_blob_times > 0)
    {
        clusters[0] = (double *)malloc(num_blob_times * sizeof(double));
        cluster_sizes[0] = 1;
        clusters[0][0] = blob_times[0];
        num_clusters = 1;

        for (int k = 1; k < num_blob_times; k++)
        {
            double delta_time = blob_times[k] - blob_times[k - 1];
            if (delta_time <= time_tolerance)
            {
                clusters[num_clusters - 1][cluster_sizes[num_clusters - 1]] = blob_times[k];
                cluster_sizes[num_clusters - 1]++;
            }
            else
            {
                clusters[num_clusters] = (double *)malloc((num_blob_times - k) * sizeof(double));
                clusters[num_clusters][0] = blob_times[k];
                cluster_sizes[num_clusters] = 1;
                num_clusters++;
            }
        }
    }

    // Calculate midpoints
    double *cluster_midpoints = (double *)malloc(num_clusters * sizeof(double));
    int count_midpoints = 0;

    for (int i = 0; i < num_clusters; i++)
    {
        double cluster_duration = clusters[i][cluster_sizes[i] - 1] - clusters[i][0];
        if (cluster_duration >= min_blob_duration)
        {
            double sum_times = 0.0;
            for (int j = 0; j < cluster_sizes[i]; j++)
            {
                sum_times += clusters[i][j];
            }
            double midpoint = sum_times / cluster_sizes[i];
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
        free(intensity_dB_filtered[i]);
        free(intensity_mp_bp[i]);
    }
    free(intensity_dB_filtered);
    free(intensity_mp_bp);
    free(frequencies_mp_bp);
    free(times_mp_bp);
    free(filtered_signal_mp);

    return cluster_midpoints;
}