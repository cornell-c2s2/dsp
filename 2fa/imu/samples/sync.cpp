// sync.cpp  — IMU gesture classification with Edge Impulse

#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>

extern "C" {
#include "lib/mpu6050.h"
}

// Edge Impulse includes
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

// Labels from Edge Impulse model
static const char *LABEL_TURN   = "turn";
static const char *LABEL_RANDOM = "random";

static constexpr int NUM_WINDOWS = 10; 

static constexpr float MIN_CONFIDENCE = 0.6f;

enum WindowLabel : int {
    WL_TURN   = 0,
    WL_RANDOM = 1,
    WL_OTHER  = 2           
};

int main() {
    stdio_init_all();
    sleep_ms(2000);

    // I2C init for MPU6050
    i2c_init(MPU6050_I2C, 400 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);

    // Reset and initialize MPU6050
    mpu6050_reset();

    printf("=== Edge Impulse IMU Gesture Classifier (Long Capture) ===\n");

    printf("Per-window config:\n");
    printf("  Window samples : %d\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    printf("  Axes per frame : %d\n", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
    printf("  Input size     : %d values\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("  Interval       : %.2f ms (%.2f Hz)\n",
           EI_CLASSIFIER_INTERVAL_MS,
           1000.0f / EI_CLASSIFIER_INTERVAL_MS);

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 6) {
        printf("WARNING: model expects %d values per frame (not 6). "
               "Check input axis mapping.\n",
               EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
    }

    // One EI window worth of data
    static float imu_window[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

    int16_t accel_raw[3];
    int16_t gyro_raw[3];
    int16_t temp_raw;

    const size_t frame_size  = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
    const size_t window_size = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    static int window_labels[NUM_WINDOWS];

    while (true) {
        // countdown
        printf("\nCountdown!\n");
        for (int c = 3; c >= 1; --c) {
            printf("%d...\n", c);
            sleep_ms(1000);
        }
        printf("Classifying %d windows...\n", NUM_WINDOWS);

        int turn_count    = 0;
        int random_count  = 0;
        int unknown_count = 0;

        // Clear labels
        for (int i = 0; i < NUM_WINDOWS; ++i) {
            window_labels[i] = WL_OTHER;  // default to "other/unknown"
        }

        for (int w = 0; w < NUM_WINDOWS; ++w) {
            // one window at a time
            for (size_t t = 0; t < window_size; ++t) {
                const size_t base = t * frame_size;

                uint64_t next_sample_time =
                    time_us_64() + (uint64_t)(EI_CLASSIFIER_INTERVAL_MS * 1000.0f);

                // Read IMU
                mpu6050_read_raw(accel_raw, gyro_raw, &temp_raw);

                // Map to model input format
                imu_window[base + 0] = static_cast<float>(accel_raw[0]); // ax
                imu_window[base + 1] = static_cast<float>(accel_raw[1]); // ay
                imu_window[base + 2] = static_cast<float>(accel_raw[2]); // az
                imu_window[base + 3] = static_cast<float>(gyro_raw[0]);  // gx
                imu_window[base + 4] = static_cast<float>(gyro_raw[1]);  // gy
                imu_window[base + 5] = static_cast<float>(gyro_raw[2]);  // gz

                // Keep sample interval
                uint64_t now = time_us_64();
                if (next_sample_time > now) {
                    sleep_us(next_sample_time - now);
                }
            }

            signal_t signal;
            int sig_err = numpy::signal_from_buffer(
                imu_window,
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
                &signal
            );
            if (sig_err != 0) {
                printf("[Window %d] ERROR: signal_from_buffer failed (%d)\n",
                       w + 1, sig_err);
                continue;
            }

            // Run inference
            ei_impulse_result_t result = { 0 };
            EI_IMPULSE_ERROR rc = run_classifier(&signal, &result, false);
            if (rc != EI_IMPULSE_OK) {
                printf("[Window %d] ERROR: run_classifier failed (%d)\n", w + 1, rc);
                continue;
            }

            // Find best label
            float best_val = 0.0f;
            const char *best_label = nullptr;

            for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; ++i) {
                const char *label = result.classification[i].label;
                float value       = result.classification[i].value;

                if (value > best_val) {
                    best_val   = value;
                    best_label = label;
                }
            }

            if (!best_label || best_val < MIN_CONFIDENCE) {
                printf("[Window %d] Low confidence or no label (max=%.3f)\n",
                       w + 1, best_val);
                unknown_count++;
                window_labels[w] = WL_OTHER;
                continue;
            }

            // Update counts + labels
            if (std::strcmp(best_label, LABEL_TURN) == 0) {
                turn_count++;
                window_labels[w] = WL_TURN;
            }
            else if (std::strcmp(best_label, LABEL_RANDOM) == 0) {
                random_count++;
                window_labels[w] = WL_RANDOM;
            }
            else {
                unknown_count++;
                window_labels[w] = WL_OTHER;
            }
        }

        // Helper lambda to print window list for a given label
        auto print_windows_for_label = [&](WindowLabel label) {
            bool first = true;
            printf(" (windows: ");
            for (int i = 0; i < NUM_WINDOWS; ++i) {
                if (window_labels[i] == label) {
                    if (!first) {
                        printf(", ");
                    }
                    printf("%d", i + 1); 
                    first = false;
                }
            }
            if (first) {
                printf("none");
            }
            printf(")\n");
        };

        // Summary
        printf("\n=== Summary over %d windows ===\n", NUM_WINDOWS);

        printf("  turn   : %d", turn_count);
        print_windows_for_label(WL_TURN);

        printf("  random : %d", random_count);
        print_windows_for_label(WL_RANDOM);

        printf("  other/unknown : %d", unknown_count);
        print_windows_for_label(WL_OTHER);

        printf("==========================================\n");

        sleep_ms(1000);
    }

    return 0;
}