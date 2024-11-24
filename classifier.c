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
bool butter_bandpass(double lowcut, double highcut, double fs, int order, double *b, double *a);
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output);
void compute_spectrogram(double *signal, int signal_length, int fs, double **frequencies, double **times, double ***intensity, int *freq_bins, int *time_bins);
double normalize_intensity(double value, double min, double max);
void find_midpoints(double *signal, int signal_length, int fs, double *frequencies, double *times, double **intensity, int freq_bins, int time_bins, double lower_threshold_dB, double min_blob_duration, double time_tolerance, double *midpoints, int *midpoint_count);
double sum_intense(double *intensity, int freq_bins, int time_bins, double *frequencies, double *times, double midpoint, double lower, double upper, double half_range);

int main()
{
    // Variables
    // char folder[] = "testing";
    char folder[] = "isolatedtest";

    struct dirent *entry;
    DIR *directory = opendir(folder);
    while ((entry = readdir(directory)) != NULL)
    {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        {

            // audioFile = "testing/2287-sj.wav"
            // samplingFreq, mySound = wavfile.read(audioFile)
            // # Normalize
            // mySound = mySound / (2.**15)
            // # If stereo, take one channel
            // mySoundOneChannel = mySound[:, 0]

            char audioFile[MAX_FILENAME];

            // snprintf(audioFile, sizeof(audioFile), "testing/%s", entry->d_name);
            snprintf(audioFile, sizeof(audioFile), "isolatedtest/%s", entry->d_name);

            bool showGraphsAndPrint = false; // Set to true to enable plotting and printing

            // Step 1: Read the WAV file
            const char *filename = "testing/2287-sj.wav";
            int16_t *wav_data = NULL;
            int samplingFreq, num_frames, num_channels;

            if (!read_wav_file(filename, &wav_data, &samplingFreq, &num_frames, &num_channels))
            {
                fprintf(stderr, "Failed to load WAV file.\n");
                return EXIT_FAILURE;
            }

            printf("Loaded WAV file: %s\n", filename);
            printf("Sample Rate: %d Hz\n", samplingFreq);
            printf("Channels: %d (output as mono)\n", num_channels);
            printf("Total Samples: %d\n", num_frames);

            // Step 2: Normalize and convert WAV data to double
            double *data = malloc(num_frames * sizeof(double));
            if (!data)
            {
                fprintf(stderr, "Memory allocation failed for data array.\n");
                free(wav_data);
                return EXIT_FAILURE;
            }
            for (int i = 0; i < num_frames; i++)
            {
                data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
            }

            free(wav_data); // No longer needed

            { // // checkpt1 passes: 'data' in C is equiv to 'mySoundOneChannel'in python
              // FILE *output_file = fopen("checkpt1.txt", "w");
              // if (output_file)
              // {
              //     for (int i = 0; i < num_frames; i++)
              //     {
              //         fprintf(output_file, "%e\n", data[i]);
              //     }
              //     fclose(output_file);
              //     printf("Filtered data saved to 'checkpt1.txt'\n");
              // }
              // else
              // {
              //     fprintf(stderr, "Failed to open file for writing.\n");
              // }
            }

            // Step 3: Apply Butterworth bandpass filter
            double *filtered = malloc(num_frames * sizeof(double));
            if (!filtered)
            {
                fprintf(stderr, "Memory allocation failed for filtered array.\n");
                free(data);
                return EXIT_FAILURE;
            }
            double lowcut = 6000.0;
            double highcut = 15000.0;
            int order = 4;       // Changed to 2 for second-order filter
            double b[9] = {0.0}; // Coefficients for second-order filter
            double a[9] = {0.0};
            butter_bandpass(lowcut, highcut, (double)samplingFreq, order, b, a);
            // Apply Butterworth bandpass filter
            double *filtered_signal = (double *)malloc(num_frames * sizeof(double));
            butter_bandpass_filter(data, num_frames, b, a, filtered_signal);

            { // checkpt2 passes: 'filtered_signal' in C is equiv to 'filtered_signal' in python
              // FILE *output_file2 = fopen("checkpt2.txt", "w");
              // if (output_file2)
              // {
              //     for (int i = 0; i < num_frames; i++)
              //     {
              //         fprintf(output_file2, "%e\n", filtered_signal[i]);
              //     }
              //     fclose(output_file2);
              //     printf("Filtered data saved to 'checkpt2.txt'\n");
              // }
              // else
              // {
              //     fprintf(stderr, "Failed to open file for writing.\n");
              // }
            }
            // free(data); // No longer needed

            // Compute spectrogram (simplified)
            double *frequencies = NULL;
            double *times = NULL;
            double **intensity = NULL;
            int freq_bins = 0, time_bins = 0;

            compute_spectrogram(data, num_frames, samplingFreq, &frequencies, &times, &intensity, &freq_bins, &time_bins);
            // Save intensity_bp to a text file
            FILE *f_intensity = fopen("checkpt3.txt", "w");
            for (int i = 0; i < freq_bins; i++)
            {
                for (int j = 0; j < time_bins; j++)
                {
                    fprintf(f_intensity, "%e ", intensity[i][j]);
                }
                fprintf(f_intensity, "\n");
            }
            fclose(f_intensity);
            printf("original intensity saved to 'checkpt3.txt'\n");

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
            lowcut = 2000.0;
            highcut = 6000.0;
            order = 4; // Changed to 2 for second-order filter

            // Get Butterworth filter coefficients
            if (!butter_bandpass(lowcut, highcut, (double)samplingFreq, order, b, a))
            {
                printf("Failed to design Butterworth filter.\n");
                free(data);
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
            double *filtered_signal2 = (double *)malloc(num_frames * sizeof(double));
            butter_bandpass_filter(data, num_frames, b, a, filtered_signal2);

            // Find midpoints
            double lower_threshold_dB = 45.0;
            double time_tolerance = 0.05;
            double min_blob_duration = 0.15;
            double *midpoints = (double *)malloc(time_bins * sizeof(double)); // Maximum possible
            int midpoint_count = 0;

            find_midpoints(filtered_signal2, num_frames, samplingFreq, frequencies, times, intensity, freq_bins, time_bins, lower_threshold_dB, min_blob_duration, time_tolerance, midpoints, &midpoint_count);

            // Calculate Spectrogram with new bandpass
            lowcut = 6000.0;
            highcut = 15000.0;
            double lower_threshold_dB_normalized = 0.85;
            double upper_threshold_dB_normalized = 0.9;

            // Design new Butterworth filter
            if (!butter_bandpass(lowcut, highcut, (double)samplingFreq, order, b, a))
            {
                printf("Failed to design Butterworth filter.\n");
                free(data);
                free(filtered_signal2);
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
            butter_bandpass_filter(data, num_frames, b, a, filtered_signal_bp);

            // Compute spectrogram of filtered_signal_bp
            double *frequencies_bp = NULL;
            double *times_bp = NULL;
            double **intensity_bp = NULL;
            int freq_bins_bp = 0, time_bins_bp = 0;

            // compute_spectrogram(filtered_signal_bp, num_frames, samplingFreq, &frequencies_bp, &times_bp, &intensity_bp, &freq_bins_bp, &time_bins_bp);
            // printf("%d\n", samplingFreq);

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
                    printf("\n", audioFile);
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
            free(data);
            free(filtered_signal);
            free(filtered_signal_bp);
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
bool butter_bandpass(double lowcut, double highcut, double fs, int order, double *b, double *a)
{
    if (order != 4)
    {
        fprintf(stderr, "Only fourth-order (order=4) Butterworth filters are implemented.\n");
        return false;
    }

    // Pre-warp the frequencies
    double nyq = 0.5 * fs;
    double low = lowcut / nyq;
    double high = highcut / nyq;

    if (lowcut == 2000 && highcut == 6000)
    {
        b[0] = 0.00021314;
        b[1] = 0.;
        b[2] = -0.00085255;
        b[3] = 0.;
        b[4] = 0.00127883;
        b[5] = 0.;
        b[6] = -0.00085255;
        b[7] = 0.;
        b[8] = 0.00021314;

        a[0] = 1.;
        a[1] = -7.12847885;
        a[2] = 22.41882266;
        a[3] = -40.62891245;
        a[4] = 46.40780141;
        a[5] = -34.21333503;
        a[6] = 15.89913237;
        a[7] = -4.25840048;
        a[8] = 0.50337536;
    }
    else if (lowcut == 6000 && highcut == 15000)
    {
        b[0] = 0.00386952;
        b[1] = 0.;
        b[2] = -0.01547807;
        b[3] = 0.;
        b[4] = 0.02321711;
        b[5] = 0.;
        b[6] = -0.01547807;
        b[7] = 0.;
        b[8] = 0.00386952;

        a[0] = 1.;
        a[1] = -5.22664543;
        a[2] = 12.83819436;
        a[3] = -19.22549589;
        a[4] = 19.15517565;
        a[5] = -12.98213646;
        a[6] = 5.84957071;
        a[7] = -1.60753218;
        a[8] = 0.2088483;
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
        for (int j = 1; j <= 9; j++)
        {
            w0 -= a[j] * w[j - 1];
        }

        double y = b[0] * w0;
        for (int j = 1; j <= 9; j++)
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
    // Parameters matching the default values in the Python function
    int window_size = 256;
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
    FILE *freqs = fopen("freqs.txt", "w");
    for (int i = 0; i < *freq_bins; i++)
    {
        fprintf(freqs, "%e\n", (*frequencies)[i]);
    }
    fclose(freqs);
    printf("freqs data saved to 'freqs.txt'\n");

    // Compute time values
    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;
        (*times)[t] = ((double)(start + window_size / 2)) / fs;
    }
    FILE *timz = fopen("timz.txt", "w");
    for (int i = 0; i < *time_bins; i++)
    {
        fprintf(timz, "%e\n", (*times)[i]);
    }
    fclose(timz);
    printf("timz data saved to 'timz.txt'\n");

    // Create the Tukey window
    double *window = (double *)malloc(window_size * sizeof(double));
    double N_minus_1 = (double)(window_size - 1);
    int M = window_size + 1; // Total number of points + 1 bc symmetric idk the python did this

    if (alpha <= 0)
    {
        // Rectangular window
        for (int i = 0; i < M; i++)
        {
            window[i] = 1.0;
        }
    }
    else if (alpha >= 1.0)
    {
        // Hann window
        for (int i = 0; i < M; i++)
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
    // // this is correct now the M was weird asf and throwing stuff off
    // FILE *winds = fopen("winds.txt", "w");
    // for (int i = 0; i < window_size; i++)
    // {
    //     fprintf(winds, "%e\n", window[i]);
    // }
    // fclose(winds);
    // printf("winds data saved to 'winds.txt'\n");

    // Compute the window power (sum of squares)
    // this U is equal to 1/scale in the python implementation
    double U = 0.0;
    for (int i = 0; i < window_size; i++)
    {
        U += window[i] * window[i];
    }
    U *= fs; // Include sampling frequency in scaling
    // printf("%f\n", U);

    // Allocate memory for FFT input and output
    double *segment = (double *)malloc(nfft * sizeof(double));
    fftw_complex *out = (fftw_complex *)fftw_malloc((*freq_bins) * sizeof(fftw_complex));

    // Create FFTW plan
    fftw_plan p = fftw_plan_dft_r2c_1d(nfft, segment, out, FFTW_ESTIMATE);

    // Process each segment
    //*time_bins is equal to n_segments in python
    // printf("%d\n", *time_bins);

    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * hop_size;

        // Extract the segment and apply zero-padding if necessary
        for (int i = 0; i < window_size; i++)
        {
            // printf("%d hastart index \n", start + i);

            if (start + i < signal_length)
                segment[i] = signal[start + i];
            else
                segment[i] = 0.0;
        }
        printf("%e \n", segment[0]);

        // Apply the window to the segment
        for (int i = 0; i < window_size; i++)
        {
            segment[i] *= window[i];
        }

        // Detrend the segment (remove mean)
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
        // // idkrn
        // FILE *segs = fopen("segs.txt", "w");
        // for (int i = 0; i < window_size; i++)
        // {
        //     fprintf(segs, "%e\n", segment[i]);
        // }
        // fclose(segs);
        // printf("segs data saved to 'segs.txt'\n");

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
        if (nfft % 2 == 0) // Even nfft
        {
            for (int f = 1; f < *freq_bins - 1; f++)
            {
                (*Sxx)[f][t] *= 2.0;
            }
        }
        else // Odd nfft
        {
            for (int f = 1; f < *freq_bins; f++)
            {
                (*Sxx)[f][t] *= 2.0;
            }
        }
    }

    // Clean up
    fftw_destroy_plan(p);
    fftw_free(out);
    free(segment);
    free(window);
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
