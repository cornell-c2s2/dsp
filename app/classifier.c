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
void spectrogram(double *signal, int signal_length, double fs,
                 int nperseg, int noverlap, int nfft,
                 double **frequencies, int *num_freqs,
                 double **times, int *num_times,
                 double ***intensity);

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

    // Spectrogram parameters
    int nperseg = 256;
    int noverlap = 128;
    int nfft = 256;

    // Output variables
    double *frequencies;
    int num_freqs;
    double *times;
    int num_times;
    double **intensity;

    // Compute spectrogram
    spectrogram(signal, signal_length, fs,
                nperseg, noverlap, nfft,
                &frequencies, &num_freqs,
                &times, &num_times,
                &intensity);

    // (Optional) Process or display the spectrogram data
    // ...

    FILE *output_file = fopen(filename, "w");
    if (output_file)
    {
        for (int i = 0; i < num_freqs; i++)
        { // Iterate over frequency bins (rows)
            for (int j = 0; j < num_times; j++)
            {                                                 // Iterate over time bins (columns)
                fprintf(output_file, "%e ", intensity[i][j]); // Write value with a space
            }
            fprintf(output_file, "\n"); // Add a newline at the end of each row
        }
        fclose(output_file);
        printf("Spectrogram data saved to '%s'\n", filename);
    }
    else
    {
        fprintf(stderr, "Failed to open file '%s' for writing.\n", filename);
    }

    // Free allocated memory
    for (int i = 0; i < num_times; i++)
    {
        free(intensity[i]);
    }
    free(intensity);
    free(frequencies);
    free(times);
    free(signal);

    return EXIT_SUCCESS;
}

void create_hann_window(double *window, int nperseg)
{
    for (int i = 0; i < nperseg; i++)
    {
        window[i] = 0.5 * (1 - cos(2 * PI * i / (nperseg - 1)));
    }
}
int calculate_num_segments(int signal_length, int nperseg, int noverlap)
{
    int step = nperseg - noverlap;
    return (signal_length - noverlap) / step;
}
void spectrogram(double *signal, int signal_length, double fs,
                 int nperseg, int noverlap, int nfft,
                 double **frequencies, int *num_freqs,
                 double **times, int *num_times,
                 double ***intensity)
{

    // Calculate the number of segments
    int step = nperseg - noverlap;
    *num_times = calculate_num_segments(signal_length, nperseg, noverlap);
    *num_freqs = nfft / 2 + 1;

    // Allocate memory for output arrays
    *frequencies = malloc(sizeof(double) * (*num_freqs));
    *times = malloc(sizeof(double) * (*num_times));
    *intensity = malloc(sizeof(double *) * (*num_times));
    for (int i = 0; i < *num_times; i++)
    {
        (*intensity)[i] = malloc(sizeof(double) * (*num_freqs));
    }

    // Create window function
    double *window = malloc(sizeof(double) * nperseg);
    create_hann_window(window, nperseg);

    // Frequency vector
    for (int i = 0; i < *num_freqs; i++)
    {
        (*frequencies)[i] = i * fs / nfft;
    }

    // Time vector
    for (int i = 0; i < *num_times; i++)
    {
        (*times)[i] = (i * step + nperseg / 2) / fs;
    }

    // FFT setup
    fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_plan plan = fftw_plan_dft_1d(nfft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Main loop over segments
    for (int i = 0; i < *num_times; i++)
    {
        // Apply window and zero-padding
        for (int j = 0; j < nperseg; j++)
        {
            in[j][0] = signal[i * step + j] * window[j]; // Real part
            in[j][1] = 0.0;                              // Imaginary part
        }
        for (int j = nperseg; j < nfft; j++)
        {
            in[j][0] = 0.0;
            in[j][1] = 0.0;
        }

        // Compute FFT
        fftw_execute(plan);

        // Compute power spectrum
        for (int j = 0; j < *num_freqs; j++)
        {
            double real = out[j][0];
            double imag = out[j][1];
            (*intensity)[i][j] = (real * real + imag * imag) / (nperseg * fs);
        }
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    free(window);
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