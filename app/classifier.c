#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>
#include <complex.h>
#include <stdio.h>

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
void spectrogram(double *x, int x_len, double fs,
                 double **freqs_out, int *freqs_len,
                 double **times_out, int *times_len,
                 double ***Sxx_out, int *Sxx_rows, int *Sxx_cols);

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
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("Normalized data[%d]: %f\n", i, data[i]);
    // }

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

    // Step 4: Calculate the spectrogram
    double fs = 1000.0; // Sampling frequency
    int signal_length = 10000;
    double *signal = malloc(sizeof(double) * signal_length);

    // Generate a test signal (e.g., a sine wave with noise)
    for (int i = 0; i < signal_length; i++)
    {
        double t = i / fs;
        signal[i] = sin(2 * PI * 50 * t) + 0.5 * sin(2 * PI * 80 * t);
    }

    // Output variables
    double *freqs;
    int num_freqs;
    double *times;
    int num_times;
    double **Sxx;
    int Sxx_rows, Sxx_cols;

    // Compute spectrogram
    spectrogram(signal, signal_length, fs, &freqs, &num_freqs, &times, &num_times, &Sxx, &Sxx_rows, &Sxx_cols);

    // (Optional) Process or display the spectrogram data
    // ...

    FILE *output_file = fopen("data/spect_c.txt", "w");
    if (output_file)
    {
        for (int i = 0; i < Sxx_rows; i++)
        { // Iterate over frequency bins (rows)
            for (int j = 0; j < Sxx_cols; j++)
            {                                           // Iterate over time bins (columns)
                fprintf(output_file, "%e ", Sxx[i][j]); // Write value with a space
            }
            fprintf(output_file, "\n"); // Add a newline at the end of each row
        }
        fclose(output_file);
        printf("Spectrogram data saved to '%s'\n", "data/spect_c.txt");
    }
    else
    {
        fprintf(stderr, "Failed to open file '%s' for writing.\n", "data/spect_c.txt");
    }

    // Free allocated memory
    free(signal);
    free(freqs);
    free(times);
    for (int i = 0; i < Sxx_rows; i++)
    {
        free(Sxx[i]);
    }
    free(Sxx);

    return EXIT_SUCCESS;
}

// Function to generate Tukey window
void tukey_window(double *win, int N, double alpha)
{
    int i;
    double M = N - 1;
    double n;

    for (i = 0; i < N; i++)
    {
        n = (double)i;
        if (n < alpha * M / 2)
        {
            win[i] = 0.5 * (1 + cos(M_PI * (2 * n / (alpha * M) - 1)));
        }
        else if (n <= M * (1 - alpha / 2))
        {
            win[i] = 1.0;
        }
        else
        {
            win[i] = 0.5 * (1 + cos(M_PI * (2 * n / (alpha * M) - 2 / alpha + 1)));
        }
    }
}

// Function to compute mean of an array
double compute_mean(double *array, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += array[i];
    }
    return sum / N;
}

// Function to compute FFT and power spectral density
void compute_fft(double *segment, int nfft, int return_onesided, double scale, int num_freqs, double *Sxx_column)
{
    fftw_complex *out;
    fftw_plan p;

    if (return_onesided)
    {
        int n_out = nfft / 2 + 1;
        out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_out);
        p = fftw_plan_dft_r2c_1d(nfft, segment, out, FFTW_ESTIMATE);
        fftw_execute(p);

        // Compute power spectral density
        for (int i = 0; i < num_freqs; i++)
        {
            double re = out[i][0];
            double im = out[i][1];
            Sxx_column[i] = (re * re + im * im) * scale;
        }

        fftw_destroy_plan(p);
        fftw_free(out);
    }
    else
    {
        // Two-sided spectrum not implemented
        fprintf(stderr, "Two-sided spectrum not implemented.\n");
        exit(1);
    }
}

void spectrogram(double *x, int x_len, double fs,
                 double **freqs_out, int *freqs_len,
                 double **times_out, int *times_len,
                 double ***Sxx_out, int *Sxx_rows, int *Sxx_cols)
{
    // Default parameters
    int nperseg = 256;              // Length of each segment
    int noverlap = nperseg / 8;     // 32 points overlap
    int nfft = nperseg;             // FFT length
    int nstep = nperseg - noverlap; // Step size between segments
    double alpha = 0.25;            // Tukey window alpha
    int return_onesided = 1;        // Return one-sided spectrum for real data

    // Generate the window function
    double *win = (double *)malloc(nperseg * sizeof(double));
    tukey_window(win, nperseg, alpha);

    // Compute the scale factor
    double win_sum_sq = 0.0;
    for (int i = 0; i < nperseg; i++)
    {
        win_sum_sq += win[i] * win[i];
    }
    double scale = 1.0 / (fs * win_sum_sq);

    // Determine the number of segments
    int n_segments = (x_len - nperseg) / nstep + 1;

    // Prepare frequency and time arrays
    int num_freqs = nfft / 2 + 1;
    *freqs_len = num_freqs;
    *freqs_out = (double *)malloc(num_freqs * sizeof(double));
    for (int i = 0; i < num_freqs; i++)
    {
        (*freqs_out)[i] = (double)i * fs / nfft;
    }

    *times_len = n_segments;
    *times_out = (double *)malloc(n_segments * sizeof(double));
    for (int i = 0; i < n_segments; i++)
    {
        (*times_out)[i] = (i * nstep) / fs + (nperseg / 2) / fs;
    }

    // Initialize the spectrogram array
    *Sxx_rows = num_freqs;
    *Sxx_cols = n_segments;
    double **Sxx = (double **)malloc(num_freqs * sizeof(double *));
    for (int i = 0; i < num_freqs; i++)
    {
        Sxx[i] = (double *)calloc(n_segments, sizeof(double));
    }

    // Loop over each segment
    for (int i = 0; i < n_segments; i++)
    {
        int start = i * nstep;
        // Extract the segment
        double *segment = (double *)malloc(nperseg * sizeof(double));
        for (int j = 0; j < nperseg; j++)
        {
            segment[j] = x[start + j];
        }

        // Detrend the segment (remove the mean)
        double mean = compute_mean(segment, nperseg);
        for (int j = 0; j < nperseg; j++)
        {
            segment[j] -= mean;
        }

        // Apply the window to the segment
        for (int j = 0; j < nperseg; j++)
        {
            segment[j] *= win[j];
        }

        // Compute the FFT and power spectral density
        double *Sxx_column = (double *)malloc(num_freqs * sizeof(double));
        compute_fft(segment, nfft, return_onesided, scale, num_freqs, Sxx_column);

        // Store the result in Sxx
        for (int k = 0; k < num_freqs; k++)
        {
            Sxx[k][i] = Sxx_column[k];
        }

        free(segment);
        free(Sxx_column);
    }

    // Adjust scaling for one-sided spectrum
    if (return_onesided)
    {
        if (nfft % 2 == 0)
        {
            for (int i = 1; i < num_freqs - 1; i++)
            {
                for (int j = 0; j < n_segments; j++)
                {
                    Sxx[i][j] *= 2.0;
                }
            }
        }
        else
        {
            for (int i = 1; i < num_freqs; i++)
            {
                for (int j = 0; j < n_segments; j++)
                {
                    Sxx[i][j] *= 2.0;
                }
            }
        }
    }

    // Assign output
    *Sxx_out = Sxx;

    // Free window
    free(win);
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