// main_test.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include "stop_detector.h"
#include "mfcc_params.h"   // MFCC_SAMPLE_RATE

// Directory containing test WAV files
#define TEST_DIR "../../data/testing"

// --- Simple WAV reader for PCM 16-bit files -----------------------------
// - Supports mono or stereo
// - Converts to float in [-1, 1]
// - If stereo, averages L/R to mono
// Returns 1 on success, 0 on failure
static int read_wav_mono_f32(const char *filename,
                             float **out_signal,
                             int *out_num_samples,
                             int expected_sample_rate)
{
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: could not open '%s'\n", filename);
        return 0;
    }

    // --- RIFF header ---
    char riff_id[4];
    uint32_t riff_size = 0;
    char wave_id[4];

    if (fread(riff_id, 1, 4, f) != 4 ||
        fread(&riff_size, 4, 1, f) != 1 ||
        fread(wave_id, 1, 4, f) != 4)
    {
        fprintf(stderr, "Error: failed to read RIFF header\n");
        fclose(f);
        return 0;
    }

    if (memcmp(riff_id, "RIFF", 4) != 0 || memcmp(wave_id, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: '%s' is not a valid RIFF/WAVE file\n", filename);
        fclose(f);
        return 0;
    }

    // Chunks we care about
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate  = 0;
    uint16_t bitsPerSample = 0;

    uint32_t data_bytes = 0;
    long data_offset = 0;

    // Chunk loop
    for (;;) {
        char chunk_id[4];
        uint32_t chunk_size = 0;

        if (fread(chunk_id, 1, 4, f) != 4) {
            fprintf(stderr, "Error: unexpected EOF while searching chunks\n");
            fclose(f);
            return 0;
        }
        if (fread(&chunk_size, 4, 1, f) != 1) {
            fprintf(stderr, "Error: failed to read chunk size\n");
            fclose(f);
            return 0;
        }

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            // Read fmt chunk
            if (chunk_size < 16) {
                fprintf(stderr, "Error: fmt chunk too small\n");
                fclose(f);
                return 0;
            }

            uint16_t blockAlign;
            uint32_t byteRate;

            if (fread(&audioFormat,   sizeof(uint16_t), 1, f) != 1 ||
                fread(&numChannels,   sizeof(uint16_t), 1, f) != 1 ||
                fread(&sampleRate,    sizeof(uint32_t), 1, f) != 1 ||
                fread(&byteRate,      sizeof(uint32_t), 1, f) != 1 ||
                fread(&blockAlign,    sizeof(uint16_t), 1, f) != 1 ||
                fread(&bitsPerSample, sizeof(uint16_t), 1, f) != 1)
            {
                fprintf(stderr, "Error: failed to read fmt chunk\n");
                fclose(f);
                return 0;
            }

            // Skip any extra fmt data
            if (chunk_size > 16) {
                if (fseek(f, chunk_size - 16, SEEK_CUR) != 0) {
                    fprintf(stderr, "Error: failed to skip extra fmt bytes\n");
                    fclose(f);
                    return 0;
                }
            }
        }
        else if (memcmp(chunk_id, "data", 4) == 0) {
            data_bytes = chunk_size;
            data_offset = ftell(f);

            if (fseek(f, chunk_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Error: failed to skip data chunk\n");
                fclose(f);
                return 0;
            }
        }
        else {
            // Unknown chunk; skip it
            if (fseek(f, chunk_size, SEEK_CUR) != 0) {
                fprintf(stderr,
                        "Error: failed to skip unknown chunk '%c%c%c%c'\n",
                        chunk_id[0], chunk_id[1], chunk_id[2], chunk_id[3]);
                fclose(f);
                return 0;
            }
        }

        // Stop if we've seen both fmt and data
        if (audioFormat != 0 && data_bytes > 0) {
            break;
        }

        // Stop if we've reached the end of RIFF
        if (ftell(f) >= (long)(riff_size + 8)) {
            break;
        }
    }

    // Basic checks
    if (audioFormat != 1) {
        fprintf(stderr, "Error: only PCM (format=1) supported, got %u\n", audioFormat);
        fclose(f);
        return 0;
    }
    if (bitsPerSample != 16) {
        fprintf(stderr, "Error: only 16-bit WAV supported, got %u bits\n", bitsPerSample);
        fclose(f);
        return 0;
    }
    if (numChannels != 1 && numChannels != 2) {
        fprintf(stderr, "Error: only mono or stereo supported, got %u channels\n", numChannels);
        fclose(f);
        return 0;
    }
    if ((int)sampleRate != expected_sample_rate) {
        fprintf(stderr,
                "Warning: WAV sampleRate=%u, expected=%d (MFCC_SAMPLE_RATE)\n",
                sampleRate, expected_sample_rate);
        // Still proceed.
    }

    if (data_bytes == 0 || data_offset == 0) {
        fprintf(stderr, "Error: data chunk not found in '%s'\n", filename);
        fclose(f);
        return 0;
    }

    // Seek to data
    if (fseek(f, data_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Error: failed to seek to data chunk\n");
        fclose(f);
        return 0;
    }

    // Number of frames and mono samples
    int bytes_per_sample = bitsPerSample / 8;
    int frame_size = bytes_per_sample * numChannels;
    int total_frames = data_bytes / frame_size;
    int num_samples_mono = total_frames;

    int16_t *tmp = (int16_t *)malloc(data_bytes);
    if (!tmp) {
        fprintf(stderr, "Error: out of memory reading WAV data\n");
        fclose(f);
        return 0;
    }

    if (fread(tmp, 1, data_bytes, f) != data_bytes) {
        fprintf(stderr, "Error: failed to read WAV data bytes\n");
        free(tmp);
        fclose(f);
        return 0;
    }

    fclose(f);

    float *signal = (float *)malloc(num_samples_mono * sizeof(float));
    if (!signal) {
        fprintf(stderr, "Error: out of memory for signal\n");
        free(tmp);
        return 0;
    }

    if (numChannels == 1) {
        for (int i = 0; i < num_samples_mono; ++i) {
            signal[i] = tmp[i] / 32768.0f;
        }
    } else { // stereo -> average
        for (int i = 0; i < num_samples_mono; ++i) {
            int16_t left  = tmp[2 * i];
            int16_t right = tmp[2 * i + 1];
            float fl = left  / 32768.0f;
            float fr = right / 32768.0f;
            signal[i] = 0.5f * (fl + fr);
        }
    }

    free(tmp);

    *out_signal = signal;
    *out_num_samples = num_samples_mono;
    return 1;
}

// --- helpers for metrics ------------------------------------------------

static int has_stop_in_name(const char *filename)
{
    // Simple case-insensitive substring search for "stop"
    const char *p = filename;
    while (*p) {
        if ((p[0] == 's' || p[0] == 'S') &&
            (p[1] == 't' || p[1] == 'T') &&
            (p[2] == 'o' || p[2] == 'O') &&
            (p[3] == 'p' || p[3] == 'P'))
        {
            return 1;
        }
        ++p;
    }
    return 0;
}

static int has_wav_extension(const char *name)
{
    const char *dot = strrchr(name, '.');
    if (!dot) return 0;
    return (strcasecmp(dot, ".wav") == 0);
}

// --- main: iterate over all files, classify, compute metrics -------------

int main(void)
{
    DIR *dir = opendir(TEST_DIR);
    if (!dir) {
        perror("opendir");
        fprintf(stderr, "Failed to open test dir: %s\n", TEST_DIR);
        return 1;
    }

    int total = 0;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    struct dirent *ent;
    printf("Scanning directory: %s\n\n", TEST_DIR);

    while ((ent = readdir(dir)) != NULL) {
        const char *name = ent->d_name;

        // Skip "." and ".."
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
            continue;
        }

        // Only process .wav files
        if (!has_wav_extension(name)) {
            continue;
        }

        char path[512];
        snprintf(path, sizeof(path), "%s/%s", TEST_DIR, name);

        float *signal = NULL;
        int num_samples = 0;

        if (!read_wav_mono_f32(path, &signal, &num_samples, MFCC_SAMPLE_RATE)) {
            fprintf(stderr, "Skipping file due to read error: %s\n", path);
            continue;
        }

        int true_label = has_stop_in_name(name) ? 1 : 0;

        float prob = classify_signal(signal, num_samples);
        int pred_label = (prob > 0.5f) ? 1 : 0;

        free(signal);

        total++;

        // Update confusion matrix
        if (true_label == 1 && pred_label == 1) tp++;
        else if (true_label == 0 && pred_label == 0) tn++;
        else if (true_label == 0 && pred_label == 1) fp++;
        else if (true_label == 1 && pred_label == 0) fn++;

        printf("File: %-40s  True: %d  Pred: %d  Prob: %.4f\n",
               name, true_label, pred_label, prob);
    }

    closedir(dir);

    printf("\n=== SUMMARY METRICS ===\n");
    printf("Total files: %d\n", total);
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n", tp, tn, fp, fn);

    if (total > 0) {
        double accuracy = (double)(tp + tn) / (double)total;
        double precision = (tp + fp) ? (double)tp / (double)(tp + fp) : 0.0;
        double recall    = (tp + fn) ? (double)tp / (double)(tp + fn) : 0.0;
        double f1        = (precision + recall) ?
                           (2.0 * precision * recall) / (precision + recall) : 0.0;

        printf("Accuracy:  %.4f\n", accuracy);
        printf("Precision: %.4f\n", precision);
        printf("Recall:    %.4f\n", recall);
        printf("F1 score:  %.4f\n", f1);
    }

    return 0;
}
