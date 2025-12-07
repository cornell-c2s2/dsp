#include "mfcc_params.h"
#include "non_stop_clip.h"
#include "pico/stdlib.h"
#include "stop_clip.h" // your baked-in test clip
#include "stop_detector.h"
#include <stdio.h>

int main(void) {
  stdio_init_all(); // enable USB stdio
  sleep_ms(2000);   // wait for USB serial to come up

  // Convert int16_t clip -> float [-1, 1]
  static float signal1[STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < STOP_CLIP_NUM_SAMPLES; ++i) {
    signal1[i] = STOP_CLIP[i] / 32768.0f;
  }

  classifier_output_t class = classify_signal(signal1, STOP_CLIP_NUM_SAMPLES);
  classifier_binary_output_t binary_class = {
    .stop = (class.stop_prob > 0.5f), 
    .speaker = (class.speaker_llr > 0.4f)
  };

  printf("Probability of 'stop' in stop clip: %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  static float signal2[NON_STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < NON_STOP_CLIP_NUM_SAMPLES; ++i) {
    signal2[i] = NON_STOP_CLIP[i] / 32768.0f;
  }

  class = classify_signal(signal2, NON_STOP_CLIP_NUM_SAMPLES);
  binary_class.stop = (class.stop_prob > 0.5f);
  binary_class.speaker = (class.speaker_llr > 0.4f);

  printf("Probability of 'stop' in non-stop clip: %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  // Keep the core alive so USB doesn’t shut down
  while (true) {
    tight_loop_contents();
  }
}
