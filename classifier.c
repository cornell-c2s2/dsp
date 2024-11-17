#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#define FRAME_SIZE 2048
#define HOP_SIZE 512
#define PI 3.14159265358979323846

typedef struct
{
    double *data;
    size_t length;
    int sample_rate;
} AudioSignal;

typedef struct
{
    double **intensity;
    double *frequencies;
    double *times;
    size_t freq_bins;
    size_t time_bins;
} Spectrogram;

void read_wav_file(const char *filename, AudioSignal *signal);
void free_audio_signal(AudioSignal *signal);
void compute_spectrogram(const AudioSignal *signal, Spectrogram *spec);
void free_spectrogram(Spectrogram *spec);
void butter_bandpass_filter(AudioSignal *signal, double lowcut, double highcut, int order);
void normalize_intensity(Spectrogram *spec);
void find_midpoints(const AudioSignal *signal, double *midpoints[], size_t *num_midpoints);
double sum_intensity(const Spectrogram *spec, double freq_low, double freq_high, double time_center, double time_half_range);
int classify_scrub_jay(const Spectrogram *spec, double *midpoints, size_t num_midpoints);

int main()
{
    DIR *d;
    struct dirent *dir;
    char folder[] = "testing";
    d = opendir(folder);
    if (!d)
    {
        fprintf(stderr, "Failed to open directory %s\n", folder);
        return 1;
    }

    while ((dir = readdir(d)) != NULL)
    {
        if (dir->d_type == DT_REG)
        {
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/%s", folder, dir->d_name);
            AudioSignal signal = {0};
            read_wav_file(filepath, &signal);

            // Normalize
            for (size_t i = 0; i < signal.length; i++)
            {
                signal.data[i] /= 32768.0;
            }

            // Band-pass filter for midpoints detection
            AudioSignal filtered_signal = signal;
            butter_bandpass_filter(&filtered_signal, 2000.0, 6000.0, 4);

            // Compute spectrogram for filtered signal
            Spectrogram spec = {0};
            compute_spectrogram(&filtered_signal, &spec);

            // Find midpoints
            double *midpoints = NULL;
            size_t num_midpoints = 0;
            find_midpoints(&filtered_signal, &midpoints, &num_midpoints);

            // Free filtered spectrogram
            free_spectrogram(&spec);

            // Band-pass filter for classification
            butter_bandpass_filter(&signal, 6000.0, 15000.0, 4);

            // Compute spectrogram for classification
            compute_spectrogram(&signal, &spec);
            normalize_intensity(&spec);

            int has_scrub_jay = classify_scrub_jay(&spec, midpoints, num_midpoints);

            if (has_scrub_jay)
            {
                printf("%s has a Scrub Jay! :)\n", dir->d_name);
            }
            else
            {
                printf("%s has no Scrub Jay! :(\n", dir->d_name);
            }

            // Cleanup
            free_audio_signal(&signal);
            free_spectrogram(&spec);
            free(midpoints);
        }
    }
    closedir(d);
    return 0;
}

void read_wav_file(const char *filename, AudioSignal *signal)
{
    SNDFILE *file;
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(SF_INFO));

    file = sf_open(filename, SFM_READ, &sfinfo);
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(1);
    }

    signal->length = sfinfo.frames;
    signal->sample_rate = sfinfo.samplerate;
    signal->data = (double *)malloc(signal->length * sizeof(double));

    if (sfinfo.channels == 1)
    {
        // Mono
        sf_read_double(file, signal->data, signal->length);
    }
    else
    {
        // Stereo or more channels, take the first channel
        double *buffer = (double *)malloc(signal->length * sfinfo.channels * sizeof(double));
        sf_read_double(file, buffer, signal->length * sfinfo.channels);
        for (size_t i = 0; i < signal->length; i++)
        {
            signal->data[i] = buffer[i * sfinfo.channels];
        }
        free(buffer);
    }
    sf_close(file);
}

void free_audio_signal(AudioSignal *signal)
{
    free(signal->data);
}

void compute_spectrogram(const AudioSignal *signal, Spectrogram *spec)
{
    size_t num_frames = (signal->length - FRAME_SIZE) / HOP_SIZE + 1;
    size_t freq_bins = FRAME_SIZE / 2 + 1;

    spec->intensity = (double **)malloc(freq_bins * sizeof(double *));
    for (size_t i = 0; i < freq_bins; i++)
    {
        spec->intensity[i] = (double *)calloc(num_frames, sizeof(double));
    }

    spec->frequencies = (double *)malloc(freq_bins * sizeof(double));
    spec->times = (double *)malloc(num_frames * sizeof(double));

    fftw_complex *out = (fftw_complex *)fftw_malloc(freq_bins * sizeof(fftw_complex));
    double *window = (double *)malloc(FRAME_SIZE * sizeof(double));
    double *frame = (double *)malloc(FRAME_SIZE * sizeof(double));
    fftw_plan plan = fftw_plan_dft_r2c_1d(FRAME_SIZE, frame, out, FFTW_ESTIMATE);

    // Generate Hann window
    for (size_t i = 0; i < FRAME_SIZE; i++)
    {
        window[i] = 0.5 * (1 - cos(2 * PI * i / (FRAME_SIZE - 1)));
    }

    for (size_t n = 0; n < num_frames; n++)
    {
        size_t offset = n * HOP_SIZE;
        for (size_t i = 0; i < FRAME_SIZE; i++)
        {
            frame[i] = signal->data[offset + i] * window[i];
        }
        fftw_execute(plan);
        for (size_t k = 0; k < freq_bins; k++)
        {
            double real = out[k][0];
            double imag = out[k][1];
            double magnitude = sqrt(real * real + imag * imag);
            spec->intensity[k][n] = 10 * log10((magnitude * magnitude) / (1e-12));
        }
        spec->times[n] = (double)(offset + FRAME_SIZE / 2) / signal->sample_rate;
    }

    for (size_t k = 0; k < freq_bins; k++)
    {
        spec->frequencies[k] = (double)k * signal->sample_rate / FRAME_SIZE;
    }

    spec->freq_bins = freq_bins;
    spec->time_bins = num_frames;

    fftw_destroy_plan(plan);
    fftw_free(out);
    free(window);
    free(frame);
}

void free_spectrogram(Spectrogram *spec)
{
    for (size_t i = 0; i < spec->freq_bins; i++)
    {
        free(spec->intensity[i]);
    }
    free(spec->intensity);
    free(spec->frequencies);
    free(spec->times);
}

void butter_bandpass(int order, double lowcut, double highcut, double fs, double *a_coeffs, double *b_coeffs)
{
    int n = order;
    double nyquist = 0.5 * fs;
    double low = lowcut / nyquist;
    double high = highcut / nyquist;

    // Pre-warp frequencies for bilinear transform
    double tan_w0 = tan(PI * (high + low) / 2.0);
    double bandwidth = tan(PI * (high - low) / 2.0);

    // Compute poles
    double *poles_real = (double *)malloc(n * sizeof(double));
    double *poles_imag = (double *)malloc(n * sizeof(double));

    for (int k = 0; k < n; k++)
    {
        double theta = PI * (2.0 * k + 1.0) / (2.0 * n);
        poles_real[k] = -bandwidth * sin(theta);
        poles_imag[k] = bandwidth * cos(theta);
    }

    // Convert poles to z-domain using bilinear transform
    double *a = (double *)calloc(n + 1, sizeof(double));
    double *b = (double *)calloc(n + 1, sizeof(double));

    a[0] = 1.0;
    b[0] = 1.0;

    for (int i = 0; i < n; i++)
    {
        double denom_real = 1.0 - 2.0 * poles_real[i] + poles_real[i] * poles_real[i] + poles_imag[i] * poles_imag[i];
        double denom_imag = 2.0 * poles_imag[i];

        double num_real = 1.0 + 2.0 * poles_real[i] + poles_real[i] * poles_real[i] + poles_imag[i] * poles_imag[i];
        double num_imag = -2.0 * poles_imag[i];

        double mag_denom = denom_real * denom_real + denom_imag * denom_imag;
        double mag_num = num_real * num_real + num_imag * num_imag;

        double gain = mag_num / mag_denom;

        // Update coefficients
        for (int j = n; j >= 1; j--)
        {
            a[j] = a[j] - 2.0 * poles_real[i] * a[j - 1] + (poles_real[i] * poles_real[i] + poles_imag[i] * poles_imag[i]) * a[j - 2];
        }
        for (int j = n; j >= 1; j--)
        {
            b[j] = b[j] - 2.0 * poles_real[i] * b[j - 1] + (poles_real[i] * poles_real[i] + poles_imag[i] * poles_imag[i]) * b[j - 2];
        }
    }

    // Normalize coefficients
    for (int i = 0; i <= n; i++)
    {
        a_coeffs[i] = a[i] / a[0];
        b_coeffs[i] = b[i] / a[0];
    }

    free(poles_real);
    free(poles_imag);
    free(a);
    free(b);
}

void butter_bandpass_filter(AudioSignal *signal, double lowcut, double highcut, int order)
{
    int n = order;
    double *a_coeffs = (double *)malloc((n + 1) * sizeof(double));
    double *b_coeffs = (double *)malloc((n + 1) * sizeof(double));

    butter_bandpass(n, lowcut, highcut, signal->sample_rate, a_coeffs, b_coeffs);

    // Apply filter using Direct Form II transposed
    double *w = (double *)calloc(n, sizeof(double));
    double *filtered_data = (double *)malloc(signal->length * sizeof(double));

    for (size_t i = 0; i < signal->length; i++)
    {
        double input = signal->data[i];
        double output = b_coeffs[0] * input + w[0];
        for (int j = 1; j < n; j++)
        {
            w[j - 1] = b_coeffs[j] * input + w[j] - a_coeffs[j] * output;
        }
        w[n - 1] = b_coeffs[n] * input - a_coeffs[n] * output;
        filtered_data[i] = output;
    }

    // Copy filtered data back to signal
    memcpy(signal->data, filtered_data, signal->length * sizeof(double));

    free(a_coeffs);
    free(b_coeffs);
    free(w);
    free(filtered_data);
}

void normalize_intensity(Spectrogram *spec)
{
    double min_intensity = spec->intensity[0][0];
    double max_intensity = spec->intensity[0][0];
    for (size_t i = 0; i < spec->freq_bins; i++)
    {
        for (size_t j = 0; j < spec->time_bins; j++)
        {
            double val = spec->intensity[i][j];
            if (!isnan(val))
            {
                if (val < min_intensity)
                    min_intensity = val;
                if (val > max_intensity)
                    max_intensity = val;
            }
        }
    }
    for (size_t i = 0; i < spec->freq_bins; i++)
    {
        for (size_t j = 0; j < spec->time_bins; j++)
        {
            if (!isnan(spec->intensity[i][j]))
            {
                spec->intensity[i][j] = (spec->intensity[i][j] - min_intensity) / (max_intensity - min_intensity);
            }
        }
    }
}

void find_midpoints(const AudioSignal *signal, double *midpoints[], size_t *num_midpoints)
{
    // Compute spectrogram
    Spectrogram spec = {0};
    compute_spectrogram(signal, &spec);

    double lower_threshold_dB = 45.0;

    // Detect time indices where intensity exceeds threshold
    double *blob_times = (double *)malloc(spec.time_bins * sizeof(double));
    size_t num_blob_times = 0;

    for (size_t t_idx = 0; t_idx < spec.time_bins; t_idx++)
    {
        int has_signal = 0;
        for (size_t f_idx = 0; f_idx < spec.freq_bins; f_idx++)
        {
            if (spec.intensity[f_idx][t_idx] > lower_threshold_dB)
            {
                has_signal = 1;
                break;
            }
        }
        if (has_signal)
        {
            blob_times[num_blob_times++] = spec.times[t_idx];
        }
    }

    // Cluster blob times
    double *cluster_midpoints = (double *)malloc(num_blob_times * sizeof(double));
    size_t num_clusters = 0;
    double *current_cluster = (double *)malloc(num_blob_times * sizeof(double));
    size_t current_cluster_size = 0;

    if (num_blob_times > 0)
    {
        current_cluster[current_cluster_size++] = blob_times[0];
        double time_tolerance = 0.05;
        double min_blob_duration = 0.15;

        for (size_t i = 1; i < num_blob_times; i++)
        {
            if (blob_times[i] - blob_times[i - 1] <= time_tolerance)
            {
                current_cluster[current_cluster_size++] = blob_times[i];
            }
            else
            {
                if (current_cluster_size > 0 && current_cluster[current_cluster_size - 1] - current_cluster[0] >= min_blob_duration)
                {
                    double sum = 0.0;
                    for (size_t j = 0; j < current_cluster_size; j++)
                    {
                        sum += current_cluster[j];
                    }
                    cluster_midpoints[num_clusters++] = sum / current_cluster_size;
                }
                current_cluster_size = 0;
                current_cluster[current_cluster_size++] = blob_times[i];
            }
        }
        if (current_cluster_size > 0 && current_cluster[current_cluster_size - 1] - current_cluster[0] >= min_blob_duration)
        {
            double sum = 0.0;
            for (size_t j = 0; j < current_cluster_size; j++)
            {
                sum += current_cluster[j];
            }
            cluster_midpoints[num_clusters++] = sum / current_cluster_size;
        }
    }

    // Set output
    *midpoints = (double *)malloc(num_clusters * sizeof(double));
    memcpy(*midpoints, cluster_midpoints, num_clusters * sizeof(double));
    *num_midpoints = num_clusters;

    // Free memory
    free(blob_times);
    free(cluster_midpoints);
    free(current_cluster);
    free_spectrogram(&spec);
}

double sum_intensity(const Spectrogram *spec, double freq_low, double freq_high, double time_center, double time_half_range)
{
    size_t freq_min_idx = 0, freq_max_idx = spec->freq_bins;
    size_t time_min_idx = 0, time_max_idx = spec->time_bins;

    // Find frequency indices
    for (size_t i = 0; i < spec->freq_bins; i++)
    {
        if (spec->frequencies[i] >= freq_low)
        {
            freq_min_idx = i;
            break;
        }
    }
    for (size_t i = freq_min_idx; i < spec->freq_bins; i++)
    {
        if (spec->frequencies[i] > freq_high)
        {
            freq_max_idx = i;
            break;
        }
    }

    // Find time indices
    for (size_t i = 0; i < spec->time_bins; i++)
    {
        if (spec->times[i] >= time_center - time_half_range)
        {
            time_min_idx = i;
            break;
        }
    }
    for (size_t i = time_min_idx; i < spec->time_bins; i++)
    {
        if (spec->times[i] > time_center + time_half_range)
        {
            time_max_idx = i;
            break;
        }
    }

    double total_intensity = 0.0;
    for (size_t i = freq_min_idx; i < freq_max_idx; i++)
    {
        for (size_t j = time_min_idx; j < time_max_idx; j++)
        {
            if (!isnan(spec->intensity[i][j]))
            {
                total_intensity += spec->intensity[i][j];
            }
        }
    }
    return total_intensity;
}

int classify_scrub_jay(const Spectrogram *spec, double *midpoints, size_t num_midpoints)
{
    int has_scrub_jay = 0;
    for (size_t idx = 0; idx < num_midpoints; idx++)
    {
        double midpoint = midpoints[idx];
        double time_threshold = 0.18;

        double above = sum_intensity(spec, 9000.0, 15000.0, midpoint, 0.18);
        double middle = sum_intensity(spec, 7000.0, 8000.0, midpoint, 0.05);
        double below = sum_intensity(spec, 1000.0, 6000.0, midpoint, 0.18);
        printf("Above: %f\n", above);
        printf("Middle: %f\n", middle);
        printf("Below: %f\n", below);
        if (middle < 50.0 && above > 200.0 && below > 200.0)
        {
            has_scrub_jay = 1;
            break;
        }
    }
    return has_scrub_jay;
}
