#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define PI 3.141592653589793

// Filter order
#define FILTER_ORDER 4
#define COEFF_SIZE (2 * FILTER_ORDER + 1)

// Precomputed coefficients for Butterworth bandpass filter
const double b[COEFF_SIZE] = {0.00386952, 0, -0.01547807, 0, 0.02321711, 0, -0.01547807, 0, 0.00386952};
const double a[COEFF_SIZE] = {1, -5.22664543, 12.83819436, -19.22549589, 19.15517565, -12.98213646, 5.84957071, -1.60753218, 0.2088483};

// Function prototypes
int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels);
void butter_bandpass_filter(const double *data, double *filtered, int data_size);
void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins);
void compute_hamming_window(double *window, int window_size);

int main()
{
    // Step 1: Read the WAV file
    const char *filename = "testing/2287-sj.wav";
    int16_t *wav_data = NULL;
    int sample_rate, num_samples, num_channels;

    if (!read_wav_file(filename, &wav_data, &sample_rate, &num_samples, &num_channels))
    {
        fprintf(stderr, "Failed to load WAV file.\n");
        return EXIT_FAILURE;
    }

    printf("Loaded WAV file: %s\n", filename);
    printf("Sample Rate: %d Hz\n", sample_rate);
    printf("Channels: %d (output as mono)\n", num_channels);
    printf("Total Samples: %d\n", num_samples);

    // Step 2: Normalize and convert WAV data to double
    double *data = malloc(num_samples * sizeof(double));
    if (!data)
    {
        fprintf(stderr, "Memory allocation failed for data array.\n");
        free(wav_data);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < num_samples; i++)
    {
        data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
    }
    for (int i = 0; i < 10; i++)
    {
        printf("Normalized data[%d]: %f\n", i, data[i]);
    }

    free(wav_data); // No longer needed

    // Step 3: Apply Butterworth bandpass filter
    double *filtered = malloc(num_samples * sizeof(double));
    if (!filtered)
    {
        fprintf(stderr, "Memory allocation failed for filtered array.\n");
        free(data);
        return EXIT_FAILURE;
    }
    butter_bandpass_filter(data, filtered, num_samples);
    free(data); // No longer needed

    // // Print the filtered data
    // printf("Filtered Data:\n");
    // for (int i = 0; i < num_samples; i++) {
    //     printf("%f\n", filtered[i]);
    // }

    // After applying Butterworth bandpass filter
    FILE *output_file = fopen("filtered_data_c.txt", "w");
    if (output_file)
    {
        for (int i = 0; i < num_samples; i++)
        {
            fprintf(output_file, "%f\n", filtered[i]);
        }
        fclose(output_file);
        printf("Filtered data saved to 'filtered_data_c.txt'\n");
    }
    else
    {
        fprintf(stderr, "Failed to open file for writing.\n");
    }

    // Step 4: Calculate the spectrogram
    double *intensity = NULL;
    double *times = NULL;
    int time_bins, freq_bins;

    int window_size = 256;
    int overlap = 128;

    spectrogram(filtered, num_samples, sample_rate, window_size, overlap, &intensity, &times, &time_bins, &freq_bins);

    printf("Spectrogram calculated: Time bins = %d, Frequency bins = %d\n", time_bins, freq_bins);

    // Print the spectrogram (example output for debugging)
    for (int i = 0; i < time_bins; i++)
    {
        printf("Time: %f\n", times[i]);
        for (int j = 0; j < freq_bins; j++)
        {
            printf("Intensity[%d][%d]: %f\n", i, j, intensity[i * freq_bins + j]);
        }
    }

    // Free allocated memory
    free(filtered);
    free(intensity);
    free(times);

    return EXIT_SUCCESS;
}

// Butterworth bandpass filter implementation
void butter_bandpass_filter(const double *data, double *filtered, int data_size)
{
    double w[FILTER_ORDER * 2] = {0}; // State variables initialized to zero

    for (int i = 0; i < data_size; i++)
    {
        double w0 = data[i];
        for (int j = 1; j <= FILTER_ORDER * 2; j++)
        {
            w0 -= a[j] * w[j - 1];
        }

        double y = b[0] * w0;
        for (int j = 1; j <= FILTER_ORDER * 2; j++)
        {
            y += b[j] * w[j - 1];
        }

        for (int j = FILTER_ORDER * 2 - 1; j > 0; j--)
        {
            w[j] = w[j - 1];
        }
        w[0] = w0;

        filtered[i] = y;
    }
}

// Spectrogram calculation
void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins)
{
    int step = window_size - overlap;
    *time_bins = (signal_size - overlap) / step;
    *freq_bins = window_size / 2 + 1; // Only half the spectrum (real-valued input)

    *intensity = malloc((*time_bins) * (*freq_bins) * sizeof(double));
    *times = malloc((*time_bins) * sizeof(double));

    double *window = malloc(window_size * sizeof(double));
    compute_hamming_window(window, window_size);

    double *x_real = malloc(window_size * sizeof(double));
    double *x_imag = calloc(window_size, sizeof(double)); // Initialize to zero

    for (int t = 0; t < *time_bins; t++)
    {
        int start = t * step;
        (*times)[t] = (double)start / fs;

        // Apply window function to segment
        for (int i = 0; i < window_size; i++)
        {
            if (start + i < signal_size)
            {
                x_real[i] = signal[start + i] * window[i];
            }
            else
            {
                x_real[i] = 0.0;
            }
        }

        // Perform DFT
        for (int k = 0; k < *freq_bins; k++)
        {
            double sum_real = 0.0;
            double sum_imag = 0.0;
            for (int n = 0; n < window_size; n++)
            {
                double angle = 2.0 * PI * k * n / window_size;
                sum_real += x_real[n] * cos(angle);
                sum_imag -= x_real[n] * sin(angle); // Note the negative sign
            }
            // Compute power spectrum
            double power = sum_real * sum_real + sum_imag * sum_imag;
            // Avoid log of zero or negative numbers
            if (power <= 0)
            {
                (*intensity)[t * (*freq_bins) + k] = -120.0; // Minimum dB value
            }
            else
            {
                (*intensity)[t * (*freq_bins) + k] = 10 * log10(power / (1e-12));
            }
        }
    }

    free(x_real);
    free(x_imag);
    free(window);
}

// Generate Hamming window
void compute_hamming_window(double *window, int window_size)
{
    for (int i = 0; i < window_size; i++)
    {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (window_size - 1));
    }
}

// Function to read WAV file and extract PCM data
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