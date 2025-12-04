#include "pico/stdlib.h"
#include "stop_detector.h"
#include "mfcc_params.h"
#include "stop_clip.h"   // your baked-in test clip
#include "non_stop_clip.h"
#include <stdio.h>

int main(void)
{
    stdio_init_all();          // enable USB stdio
    sleep_ms(2000);            // wait for USB serial to come up

    float prob;
    int detected;
    // Convert int16_t clip -> float [-1, 1]
    static float signal1[STOP_CLIP_NUM_SAMPLES];
    for (int i = 0; i < STOP_CLIP_NUM_SAMPLES; ++i) {
        signal1[i] = STOP_CLIP[i] / 32768.0f;
    }

    prob = classify_signal(signal1, STOP_CLIP_NUM_SAMPLES);
    detected = (prob > 0.5f);

    printf("Probability of 'stop' in stop clip: %f\r\n", prob);
    printf("Detected: %d\r\n", detected);

    static float signal2[NON_STOP_CLIP_NUM_SAMPLES];
    for (int i = 0; i < NON_STOP_CLIP_NUM_SAMPLES; ++i) {
        signal2[i] = NON_STOP_CLIP[i] / 32768.0f;
    }

    prob = classify_signal(signal2, NON_STOP_CLIP_NUM_SAMPLES);
    detected = (prob > 0.5f);

    printf("Probability of 'stop' in non-stop clip: %f\r\n", prob);
    printf("Detected: %d\r\n", detected);

    // Keep the core alive so USB doesn’t shut down
    while (true) {
        tight_loop_contents();
    }
}
