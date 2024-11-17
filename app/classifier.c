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
const double b[COEFF_SIZE] = {0.01775506, 0, -0.07102025, 0, 0.10653037, 0, -0.07102025, 0, 0.01775506};
const double a[COEFF_SIZE] = {1, -3.326554, 5.025296, -4.664456, 2.844493, -1.157439, 0.298745, -0.043421, 0.002888};

// Function prototypes
int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels);
void butter_bandpass_filter(const double *data, double *filtered, int data_size);
void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins);
void compute_hamming_window(double *window, int window_size);

int main() {
    // Step 1: Read the WAV file
    const char *filename = "testing/2287-sj.wav";
    int16_t *wav_data = NULL;
    int sample_rate, num_samples, num_channels;

    if (!read_wav_file(filename, &wav_data, &sample_rate, &num_samples, &num_channels)) {
        fprintf(stderr, "Failed to load WAV file.\n");
        return EXIT_FAILURE;
    }

    printf("Loaded WAV file: %s\n", filename);
    printf("Sample Rate: %d Hz\n", sample_rate);
    printf("Channels: %d (output as mono)\n", num_channels);
    printf("Total Samples: %d\n", num_samples);

    // Step 2: Normalize and convert WAV data to double
    double *data = malloc(num_samples * sizeof(double));
    if (!data) {
        fprintf(stderr, "Memory allocation failed for data array.\n");
        free(wav_data);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < num_samples; i++) {
        data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
    }
    free(wav_data); // No longer needed

    // Step 3: Apply Butterworth bandpass filter
    double *filtered = malloc(num_samples * sizeof(double));
    if (!filtered) {
        fprintf(stderr, "Memory allocation failed for filtered array.\n");
        free(data);
        return EXIT_FAILURE;
    }
    butter_bandpass_filter(data, filtered, num_samples);
    free(data); // No longer needed

    // Step 4: Calculate the spectrogram
    double *intensity = NULL;
    double *times = NULL;
    int time_bins, freq_bins;

    int window_size = 256;
    int overlap = 128;

    spectrogram(filtered, num_samples, sample_rate, window_size, overlap, &intensity, &times, &time_bins, &freq_bins);

    printf("Spectrogram calculated: Time bins = %d, Frequency bins = %d\n", time_bins, freq_bins);

    // Print the spectrogram (example output for debugging)
    for (int i = 0; i < time_bins; i++) {
        printf("Time: %f\n", times[i]);
        for (int j = 0; j < freq_bins; j++) {
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
void butter_bandpass_filter(const double *data, double *filtered, int data_size) {
    double w[FILTER_ORDER * 2] = {0}; // State variables initialized to zero

    for (int i = 0; i < data_size; i++) {
        double w0 = data[i];
        for (int j = 1; j <= FILTER_ORDER * 2; j++) {
            w0 -= a[j] * w[j - 1];
        }

        double y = b[0] * w0;
        for (int j = 1; j <= FILTER_ORDER * 2; j++) {
            y += b[j] * w[j - 1];
        }

        for (int j = FILTER_ORDER * 2 - 1; j > 0; j--) {
            w[j] = w[j - 1];
        }
        w[0] = w0;

        filtered[i] = y;
    }
}

// Spectrogram calculation
void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins) {
    int step = window_size - overlap;
    *time_bins = (signal_size - overlap) / step;
    *freq_bins = window_size / 2 + 1; // Only half the spectrum (real-valued input)

    *intensity = malloc((*time_bins) * (*freq_bins) * sizeof(double));
    *times = malloc((*time_bins) * sizeof(double));

    double *window = malloc(window_size * sizeof(double));
    compute_hamming_window(window, window_size);

    double *real = malloc(window_size * sizeof(double));
    double *imag = malloc(window_size * sizeof(double));

    for (int t = 0; t < *time_bins; t++) {
        int start = t * step;
        (*times)[t] = (double)start / fs;

        // Apply window function to segment
        for (int i = 0; i < window_size; i++) {
            if (start + i < signal_size) {
                real[i] = signal[start + i] * window[i];
            } else {
                real[i] = 0.0;
            }
            imag[i] = 0.0; // Initialize imaginary part to 0
        }

        // Perform FFT (manual implementation of DFT)
        for (int k = 0; k < window_size; k++) {
            double sum_real = 0.0;
            double sum_imag = 0.0;
            for (int n = 0; n < window_size; n++) {
                double angle = -2.0 * PI * k * n / window_size;
                sum_real += real[n] * cos(angle) - imag[n] * sin(angle);
                sum_imag += real[n] * sin(angle) + imag[n] * cos(angle);
            }
            real[k] = sum_real;
            imag[k] = sum_imag;
        }

        // Compute intensity
        for (int k = 0; k < *freq_bins; k++) {
            double power = real[k] * real[k] + imag[k] * imag[k];
            if (power <= 0) {
                (*intensity)[t * (*freq_bins) + k] = -120.0; // Minimum dB value
            } else {
                (*intensity)[t * (*freq_bins) + k] = 10 * log10(power / (10e-12));
            }
        }
    }

    free(real);
    free(imag);
    free(window);
}


// Generate Hamming window
void compute_hamming_window(double *window, int window_size) {
    for (int i = 0; i < window_size; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (window_size - 1));
    }
}


// Function to read WAV file and extract PCM data
int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
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
    if (strncmp(chunk_id, "RIFF", (size_t)4) != 0) {
        fprintf(stderr, "Not a valid RIFF file.\n");
        fclose(file);
        return 0;
    }
    fread(&chunk_size, 4, 1, file); // Chunk size
    fread(chunk_id, 1, 4, file);    // "WAVE"
    if (strncmp(chunk_id, "WAVE", (size_t)4) != 0) {
        fprintf(stderr, "Not a valid WAVE file.\n");
        fclose(file);
        return 0;
    }

    // Read fmt subchunk
    fread(chunk_id, 1, 4, file); // "fmt "
    if (strncmp(chunk_id, "fmt ", (size_t)4) != 0) {
        fprintf(stderr, "Missing 'fmt ' subchunk.\n");
        fclose(file);
        return 0;
    }
    fread(&chunk_size, 4, 1, file);       // Subchunk1Size
    fread(&audio_format, 2, 1, file);    // AudioFormat
    fread(&channels, 2, 1, file);        // NumChannels
    fread(&sample_rate_local, 4, 1, file); // SampleRate
    fread(&byte_rate, 4, 1, file);       // ByteRate
    fread(&block_align, 2, 1, file);     // BlockAlign
    fread(&bits_per_sample, 2, 1, file); // BitsPerSample

    if (audio_format != 1) {
        fprintf(stderr, "Unsupported audio format: %d\n", audio_format);
        fclose(file);
        return 0;
    }

    // Read data subchunk
    fread(chunk_id, 1, 4, file); // "data"
    while (strncmp(chunk_id, "data", (size_t)4) != 0) {
        fread(&chunk_size, 4, 1, file);
        fseek(file, chunk_size, SEEK_CUR);
        fread(chunk_id, 1, 4, file);
    }
    fread(&data_chunk_size, 4, 1, file);

    // Allocate memory for audio data
    int total_samples = data_chunk_size / (bits_per_sample / 8);
    *data = (int16_t *)malloc(total_samples * sizeof(int16_t));
    if (!*data) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        return 0;
    }
    fread(*data, sizeof(int16_t), total_samples, file);

    // Set metadata
    *sample_rate = sample_rate_local;
    *num_samples = total_samples;
    *num_channels = channels;

    fclose(file);
    return 1;
}


// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <math.h>
// #include <string.h> // For strncmp
// #include <fftw3.h>

// // Constants
// #define FILTER_ORDER 4
// #define COEFF_SIZE (2 * FILTER_ORDER + 1)

// // Precomputed coefficients for 4th-order Butterworth bandpass filter
// const double b[COEFF_SIZE] = {
//     0.00386952, 0, -0.01547807, 0, 0.02321711, 0, -0.01547807, 0, 0.00386952
// };
// const double a[COEFF_SIZE] = {
//   1, -5.22664543,  12.83819436, -19.22549589,  19.15517565,
//  -12.98213646 ,  5.84957071,  -1.60753218,   0.2088483 
// };

// // Function prototypes
// int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels);
// void butter_bandpass_filter(const double *data, double *filtered, int data_size);

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// #define PI 3.141592653589793

// // Function prototypes
// void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins);
// void compute_hamming_window(double *window, int window_size);

// int main() {
//     int signal_size = 1000;      // Example signal size
//     double fs = 44100.0;         // Sampling frequency
//     double *signal = malloc(signal_size * sizeof(double));

//     // Generate an example sine wave
//     for (int i = 0; i < signal_size; i++) {
//         signal[i] = sin(2 * PI * 1000.0 * i / fs); // 1 kHz sine wave
//     }

//     int window_size = 256;
//     int overlap = 128;
//     double *intensity = NULL;
//     double *times = NULL;
//     int time_bins, freq_bins;

//     spectrogram(signal, signal_size, fs, window_size, overlap, &intensity, &times, &time_bins, &freq_bins);

//     printf("Spectrogram calculated: Time bins = %d, Frequency bins = %d\n", time_bins, freq_bins);
//     for (int i = 0; i < time_bins; i++) {
//         printf("Time: %f\n", times[i]);
//         for (int j = 0; j < freq_bins; j++) {
//             printf("Intensity[%d][%d]: %f\n", i, j, intensity[i * freq_bins + j]);
//         }
//     }

//     free(signal);
//     free(intensity);
//     free(times);

//     return 0;
// }

// // Spectrogram calculation
// void spectrogram(const double *signal, int signal_size, double fs, int window_size, int overlap, double **intensity, double **times, int *time_bins, int *freq_bins) {
//     int step = window_size - overlap;
//     *time_bins = (signal_size - overlap) / step;
//     *freq_bins = window_size / 2 + 1;

//     *intensity = malloc((*time_bins) * (*freq_bins) * sizeof(double));
//     *times = malloc((*time_bins) * sizeof(double));

//     double *window = malloc(window_size * sizeof(double));
//     compute_hamming_window(window, window_size);

//     double *windowed_signal = malloc(window_size * sizeof(double));
//     double *real = malloc(window_size * sizeof(double));
//     double *imag = malloc(window_size * sizeof(double));

//     for (int t = 0; t < *time_bins; t++) {
//         int start = t * step;
//         (*times)[t] = (double)start / fs;

//         // Apply window function
//         for (int i = 0; i < window_size; i++) {
//             if (start + i < signal_size) {
//                 windowed_signal[i] = signal[start + i] * window[i];
//             } else {
//                 windowed_signal[i] = 0.0;
//             }
//         }

//         // Compute FFT (simple implementation)
//         for (int k = 0; k < window_size; k++) {
//             real[k] = 0.0;
//             imag[k] = 0.0;
//             for (int n = 0; n < window_size; n++) {
//                 double angle = 2 * PI * k * n / window_size;
//                 real[k] += windowed_signal[n] * cos(angle);
//                 imag[k] -= windowed_signal[n] * sin(angle);
//             }
//         }

//         // Compute power spectrum
//         for (int k = 0; k < *freq_bins; k++) {
//             (*intensity)[t * (*freq_bins) + k] = 10 * log10((real[k] * real[k] + imag[k] * imag[k]) / (10e-12));
//         }
//     }

//     free(window);
//     free(windowed_signal);
//     free(real);
//     free(imag);
// }

// // Generate Hamming window
// void compute_hamming_window(double *window, int window_size) {
//     for (int i = 0; i < window_size; i++) {
//         window[i] = 0.54 - 0.46 * cos(2 * PI * i / (window_size - 1));
//     }
// }



// // int main() {
// //     const char *filename = "testing/2287-sj.wav";
// //     int16_t *wav_data = NULL;
// //     int sample_rate, num_samples, num_channels;

// //     // Read WAV file
// //     if (read_wav_file(filename, &wav_data, &sample_rate, &num_samples, &num_channels)) {
// //         printf("Loaded WAV file: %s\n", filename);
// //         printf("Sample Rate: %d Hz\n", sample_rate);
// //         printf("Channels: %d (output as mono)\n", num_channels);
// //         printf("Total Samples: %d\n", num_samples);
// //     } else {
// //         fprintf(stderr, "Failed to load WAV file.\n");
// //         return EXIT_FAILURE;
// //     }

// //     // Convert int16_t WAV data to double array
// //    double *data = NULL; 
// //     data = (double *)malloc(num_samples * sizeof(double));
// //     if (!data) {
// //     fprintf(stderr, "Memory allocation failed for data array.\n");
// //     free(wav_data);
// //     return EXIT_FAILURE;
// //     }


// //     for (int i = 0; i < num_samples; i++) {
// //         data[i] = wav_data[i] / 32768.0; // Normalize to [-1, 1]
// //     }

// //     free(wav_data); // Free the raw WAV data

// //     // Allocate space for the filtered signal
// //     double *filtered = (double *)malloc(num_samples * sizeof(double));
// //     if (!filtered) {
// //         fprintf(stderr, "Memory allocation failed for filtered array.\n");
// //         free(data);
// //         return EXIT_FAILURE;
// //     }

// //     // Apply Butterworth bandpass filter
// //     butter_bandpass_filter(data, filtered, num_samples);

// //     // Print the filtered signal
// //     printf("Filtered Signal:\n");
// //     for (int i = 0; i < num_samples; i++) {
// //         printf("%f\n", filtered[i]);
// //     }

// //     // Free allocated memory
// //     free(data);
// //     free(filtered);

// //     return EXIT_SUCCESS;
// // }

// // Apply Butterworth bandpass filter using Direct Form II Transposed
// void butter_bandpass_filter(const double *data, double *filtered, int data_size) {
//     double w[FILTER_ORDER * 2] = {0}; // State variables initialized to zero
//     for (int i = 0; i < data_size; i++) {
//         if (isnan(data[i]) || isinf(data[i])) {
//             fprintf(stderr, "NaN or Inf detected in (og) input at index %d: %f\n", i, data[i]);
//             exit(EXIT_FAILURE);
//         }
//     }

//     for (int i = 0; i < data_size; i++) {
//         // Compute w[0]
//         double w0 = data[i];
//         for (int j = 1; j <= FILTER_ORDER * 2; j++) {
//             w0 -= a[j] * w[j - 1];
//         }

//         // Compute filtered output
//         double y = b[0] * w0;
//         for (int j = 1; j <= FILTER_ORDER * 2; j++) {
//             y += b[j] * w[j - 1];
//         }

//         // Update state variables
//         for (int j = FILTER_ORDER * 2 - 1; j > 0; j--) {
//             w[j] = w[j - 1];
//         }
//         w[0] = w0;

//         // Save the filtered output
//         filtered[i] = y;

//         // Check for NaN or Inf in the filtered output
//         if (isnan(filtered[i]) || isinf(filtered[i])) {
//             fprintf(stderr, "NaN or Inf detected in filtered output at index %d\n", i);
//             exit(EXIT_FAILURE);
//         }
//     }
// }

// // Function to read WAV file and extract PCM data
// int read_wav_file(const char *filename, int16_t **data, int *sample_rate, int *num_samples, int *num_channels) {
//     FILE *file = fopen(filename, "rb");
//     if (!file) {
//         perror("Error opening file");
//         return 0;
//     }

//     char chunk_id[4];
//     uint32_t chunk_size;
//     uint16_t audio_format, channels;
//     uint32_t sample_rate_local, byte_rate;
//     uint16_t block_align, bits_per_sample;
//     uint32_t data_chunk_size;

//     // Read RIFF header
//     fread(chunk_id, 1, 4, file); // "RIFF"
//     if (strncmp(chunk_id, "RIFF", (size_t)4) != 0) {
//         fprintf(stderr, "Not a valid RIFF file.\n");
//         fclose(file);
//         return 0;
//     }
//     fread(&chunk_size, 4, 1, file); // Chunk size
//     fread(chunk_id, 1, 4, file);    // "WAVE"
//     if (strncmp(chunk_id, "WAVE", (size_t)4) != 0) {
//         fprintf(stderr, "Not a valid WAVE file.\n");
//         fclose(file);
//         return 0;
//     }

//     // Read fmt subchunk
//     fread(chunk_id, 1, 4, file); // "fmt "
//     if (strncmp(chunk_id, "fmt ", (size_t)4) != 0) {
//         fprintf(stderr, "Missing 'fmt ' subchunk.\n");
//         fclose(file);
//         return 0;
//     }
//     fread(&chunk_size, 4, 1, file);       // Subchunk1Size
//     fread(&audio_format, 2, 1, file);    // AudioFormat
//     fread(&channels, 2, 1, file);        // NumChannels
//     fread(&sample_rate_local, 4, 1, file); // SampleRate
//     fread(&byte_rate, 4, 1, file);       // ByteRate
//     fread(&block_align, 2, 1, file);     // BlockAlign
//     fread(&bits_per_sample, 2, 1, file); // BitsPerSample

//     if (audio_format != 1) {
//         fprintf(stderr, "Unsupported audio format: %d\n", audio_format);
//         fclose(file);
//         return 0;
//     }

//     // Read data subchunk
//     fread(chunk_id, 1, 4, file); // "data"
//     while (strncmp(chunk_id, "data", (size_t)4) != 0) {
//         fread(&chunk_size, 4, 1, file);
//         fseek(file, chunk_size, SEEK_CUR);
//         fread(chunk_id, 1, 4, file);
//     }
//     fread(&data_chunk_size, 4, 1, file);

//     // Allocate memory for audio data
//     int total_samples = data_chunk_size / (bits_per_sample / 8);
//     *data = (int16_t *)malloc(total_samples * sizeof(int16_t));
//     if (!*data) {
//         fprintf(stderr, "Memory allocation failed.\n");
//         fclose(file);
//         return 0;
//     }
//     fread(*data, sizeof(int16_t), total_samples, file);

//     // Set metadata
//     *sample_rate = sample_rate_local;
//     *num_samples = total_samples;
//     *num_channels = channels;

//     fclose(file);
//     return 1;
// }

