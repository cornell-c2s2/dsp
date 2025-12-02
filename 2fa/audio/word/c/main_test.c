// #include <stdio.h>

// #include "audio_classifier_inference.h"
// #include "test_mfcc.h"

// int INPUT_SIZE = 13000;
// int main(void) {
//     if (TEST_MFCC_SIZE != INPUT_SIZE) {
//         printf("Error: TEST_MFCC_SIZE (%d) != INPUT_SIZE (%d)\n",
//                TEST_MFCC_SIZE, INPUT_SIZE);
//         return 1;
//     }

//     float prob = audio_classifier_predict(TEST_MFCC);
//     int label = audio_classifier_is_stop(TEST_MFCC, 0.5f);

//     printf("Probability of 'stop': %f\n", prob);
//     printf("Predicted label: %d\n", label);

//     return 0;
// }
// main_test.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "stop_detector.h"
#include "mfcc_params.h"   // for MFCC_SAMPLE_RATE

// Simple WAV reader for PCM 16-bit files
// - Supports mono or stereo
// - Converts to float in [-1, 1]
// - If stereo, averages channels to mono
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

    // We need to find the "fmt " chunk and the "data" chunk
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
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
            // Skip over data for now; we'll come back
            if (fseek(f, chunk_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Error: failed to skip data chunk\n");
                fclose(f);
                return 0;
            }
        }
        else {
            // Unknown chunk: just skip it
            if (fseek(f, chunk_size, SEEK_CUR) != 0) {
                fprintf(stderr, "Error: failed to skip unknown chunk '%c%c%c%c'\n",
                        chunk_id[0], chunk_id[1], chunk_id[2], chunk_id[3]);
                fclose(f);
                return 0;
            }
        }

        // Stop if we've seen both fmt and data
        if (audioFormat != 0 && data_bytes > 0) {
            break;
        }

        // If we reach end of file, break
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
        // We can still proceed if you want; for now we just warn.
    }

    if (data_bytes == 0 || data_offset == 0) {
        fprintf(stderr, "Error: data chunk not found in '%s'\n", filename);
        fclose(f);
        return 0;
    }

    // Go to data
    if (fseek(f, data_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Error: failed to seek to data chunk\n");
        fclose(f);
        return 0;
    }

    // Number of samples per channel
    int bytes_per_sample = bitsPerSample / 8;
    int frame_size = bytes_per_sample * numChannels;
    int total_frames = data_bytes / frame_size;
    int num_samples_mono = total_frames;  // one mono sample per frame

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

    // Allocate float buffer for mono signal
    float *signal = (float *)malloc(num_samples_mono * sizeof(float));
    if (!signal) {
        fprintf(stderr, "Error: out of memory for signal\n");
        free(tmp);
        return 0;
    }

    // Convert to float [-1, 1]. If stereo, average L/R.
    if (numChannels == 1) {
        for (int i = 0; i < num_samples_mono; ++i) {
            signal[i] = tmp[i] / 32768.0f;
        }
    } else { // stereo
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

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.wav>\n", argv[0]);
        return 1;
    }

    const char *wav_path = argv[1];
    float *signal = NULL;
    int num_samples = 0;

    if (!read_wav_mono_f32(wav_path, &signal, &num_samples, MFCC_SAMPLE_RATE)) {
        fprintf(stderr, "Failed to load WAV file '%s'\n", wav_path);
        return 1;
    }

    printf("Loaded '%s': %d samples at ~%d Hz\n",
           wav_path, num_samples, MFCC_SAMPLE_RATE);

    // Run the full pipeline: MFCC -> neural net
    float prob = classify_signal(signal, num_samples);
    int is_stop = classify_signal_binary(signal, num_samples, 0.5f);

    printf("Probability of 'stop': %f\n", prob);
    printf("Predicted label (threshold 0.5): %d\n", is_stop);

    free(signal);
    return 0;
}
