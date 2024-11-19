#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>

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
// void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins);
// void compute_hamming_window(double *window, int window_size);

// int main()
// {
//     // Step 1: Read the WAV file
//     const char *filename = "testing/2287-sj.wav";
//     int16_t *wav_data = NULL;
//     int sample_rate, num_samples, num_channels;

//     if (!read_wav_file(filename, &wav_data, &sample_rate, &num_samples, &num_channels))
//     {
//         fprintf(stderr, "Failed to load WAV file.\n");
//         return EXIT_FAILURE;
//     }

//     printf("Loaded WAV file: %s\n", filename);
//     printf("Sample Rate: %d Hz\n", sample_rate);
//     printf("Channels: %d (output as mono)\n", num_channels);
//     printf("Total Samples: %d\n", num_samples);

//     // Step 2: Normalize and convert WAV data to double
//     double *data = malloc(num_samples * sizeof(double));
//     if (!data)
//     {
//         fprintf(stderr, "Memory allocation failed for data array.\n");
//         free(wav_data);
//         return EXIT_FAILURE;
//     }
//     for (int i = 0; i < num_samples; i++)
//     {
//         data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
//     }
//     for (int i = 0; i < 10; i++)
//     {
//         printf("Normalized data[%d]: %f\n", i, data[i]);
//     }

//     free(wav_data); // No longer needed

//     // Step 3: Apply Butterworth bandpass filter
//     double *filtered = malloc(num_samples * sizeof(double));
//     if (!filtered)
//     {
//         fprintf(stderr, "Memory allocation failed for filtered array.\n");
//         free(data);
//         return EXIT_FAILURE;
//     }
//     butter_bandpass_filter(data, filtered, num_samples);
//     free(data); // No longer needed

//     // // Print the filtered data
//     // printf("Filtered Data:\n");
//     // for (int i = 0; i < num_samples; i++) {
//     //     printf("%f\n", filtered[i]);
//     // }

//     // After applying Butterworth bandpass filter
//     FILE *output_file = fopen("filtered_data_c.txt", "w");
//     if (output_file)
//     {
//         for (int i = 0; i < num_samples; i++)
//         {
//             fprintf(output_file, "%f\n", filtered[i]);
//         }
//         fclose(output_file);
//         printf("Filtered data saved to 'filtered_data_c.txt'\n");
//     }
//     else
//     {
//         fprintf(stderr, "Failed to open file for writing.\n");
//     }

//     // Step 4: Calculate the spectrogram
//     double *intensity = NULL;
//     double *times = NULL;
//     int time_bins, freq_bins;

//     int window_size = 256;
//     int overlap = 128;

//     spectrogram(filtered, num_samples, sample_rate, window_size, overlap, &intensity, &times, &time_bins, &freq_bins);

//     printf("Spectrogram calculated: Time bins = %d, Frequency bins = %d\n", time_bins, freq_bins);

//     // Print the spectrogram (example output for debugging)
//     // for (int i = 0; i < time_bins; i++)
//     // {
//     //     printf("Time: %f\n", times[i]);
//     //     for (int j = 0; j < freq_bins; j++)
//     //     {
//     //         printf("Intensity[%d][%d]: %e\n", i, j, intensity[i * freq_bins + j]);
//     //     }
//     // }

//     FILE *output_file2 = fopen("filtered_intensity_c.txt", "w");
//     if (output_file2)
//     {
//         for (int i = 0; i < freq_bins; i++) // Iterate over frequency bins (rows)
//         {
//             for (int j = 0; j < time_bins; j++) // Iterate over time bins (columns)
//             {
//                 fprintf(output_file2, "%e ", intensity[i * time_bins + j]); // Write value with a space
//             }
//             fprintf(output_file2, "\n"); // Add a newline at the end of each row
//         }
//         fclose(output_file2);
//         printf("Filtered data saved to 'filtered_intensity_c.txt'\n");
//     }
//     else
//     {
//         fprintf(stderr, "Failed to open file for writing.\n");
//     }

//     // Free allocated memory
//     free(filtered);
//     free(intensity);
//     free(times);

//     return EXIT_SUCCESS;
// }

void tukey_window(double *window, int N, double alpha)
{
    int n;
    int edge = (int)(alpha * (N - 1) / 2);
    for (n = 0; n < N; n++)
    {
        if (n < edge)
        {
            window[n] = 0.5 * (1 + cos(M_PI * ((2.0 * n) / (alpha * (N - 1)) - 1)));
        }
        else if (n > N - edge - 1)
        {
            window[n] = 0.5 * (1 + cos(M_PI * ((2.0 * n) / (alpha * (N - 1)) - (2 / alpha) + 1)));
        }
        else
        {
            window[n] = 1.0;
        }
    }
}
void detrend(double *data, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += data[i];
    }
    double mean = sum / N;
    for (int i = 0; i < N; i++)
    {
        data[i] -= mean;
    }
}
void spectrogram(double *x, int x_len, double fs, const char *window_type, int nperseg, int noverlap, int nfft, int detrend_flag, int return_onesided, const char *scaling, const char *mode, double **frequencies, int *freq_len, double **times, int *time_len, double ***spectrogram)
{
    int step = nperseg - noverlap;
    int num_segments = (x_len - noverlap) / step;
    *time_len = num_segments;
    *freq_len = nfft / 2 + 1;
    *frequencies = (double *)malloc((*freq_len) * sizeof(double));
    *times = (double *)malloc((*time_len) * sizeof(double));
    *spectrogram = (double **)malloc((*freq_len) * sizeof(double *));
    for (int i = 0; i < *freq_len; i++)
    {
        (*spectrogram)[i] = (double *)malloc((*time_len) * sizeof(double));
    }

    // Generate frequency vector
    for (int i = 0; i < *freq_len; i++)
    {
        (*frequencies)[i] = i * fs / nfft;
    }

    // Generate time vector
    for (int i = 0; i < num_segments; i++)
    {
        (*times)[i] = (i * step + nperseg / 2) / fs;
    }

    // Generate window
    double *window = (double *)malloc(nperseg * sizeof(double));
    if (strcmp(window_type, "tukey") == 0)
    {
        tukey_window(window, nperseg, 0.25);
    }
    else
    {
        // Implement other window types as needed
    }

    // FFTW plan
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_plan plan = fftw_plan_dft_1d(nfft, in, out, FFTW_FORWARD, FFTW_MEASURE);

    // Loop over segments
    for (int i = 0; i < num_segments; i++)
    {
        int start = i * step;
        // Copy segment and apply window
        for (int j = 0; j < nperseg; j++)
        {
            if (start + j < x_len)
            {
                in[j][0] = x[start + j] * window[j]; // Real part
                in[j][1] = 0.0;                      // Imaginary part
            }
            else
            {
                in[j][0] = 0.0;
                in[j][1] = 0.0;
            }
        }
        // Zero-padding if nfft > nperseg
        for (int j = nperseg; j < nfft; j++)
        {
            in[j][0] = 0.0;
            in[j][1] = 0.0;
        }

        // Detrend if needed
        if (detrend_flag)
        {
            double segment_mean = 0.0;
            for (int j = 0; j < nperseg; j++)
            {
                segment_mean += in[j][0];
            }
            segment_mean /= nperseg;
            for (int j = 0; j < nperseg; j++)
            {
                in[j][0] -= segment_mean;
            }
        }

        // Compute FFT
        fftw_execute(plan);

        // Compute spectrogram column
        for (int j = 0; j < *freq_len; j++)
        {
            double real = out[j][0];
            double imag = out[j][1];
            double magnitude = sqrt(real * real + imag * imag);

            if (strcmp(mode, "psd") == 0)
            {
                // Power Spectral Density
                (*spectrogram)[j][i] = (magnitude * magnitude) / (fs * nperseg);
            }
            else if (strcmp(mode, "magnitude") == 0)
            {
                (*spectrogram)[j][i] = magnitude;
            }
            // Add other modes as needed
        }
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    free(window);
}
int main()
{
    // Example signal
    int x_len = 1024;
    double x[x_len];
    double fs = 1000.0; // Sampling frequency
    for (int i = 0; i < x_len; i++)
    {
        x[i] = sin(2 * M_PI * 50 * i / fs); // 50 Hz sine wave
    }

    // Spectrogram parameters
    int nperseg = 256;
    int noverlap = 128;
    int nfft = 256;

    // Outputs
    double *frequencies;
    int freq_len;
    double *times;
    int time_len;
    double **Sxx;

    // Compute spectrogram
    spectrogram(x, x_len, fs, "tukey", nperseg, noverlap, nfft, 1, 1, "density", "psd", &frequencies, &freq_len, &times, &time_len, &Sxx);

    // Do something with the spectrogram (e.g., print or plot)

    // Clean up
    for (int i = 0; i < freq_len; i++)
    {
        free(Sxx[i]);
    }
    free(Sxx);
    free(frequencies);
    free(times);

    return 0;
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