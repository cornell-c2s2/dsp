#include "mfcc_params.h"
#include "pico/stdlib.h"
#include "stop_detector.h"
#include <stdio.h>

#include "audio/non_stop_clip.h"
#include "audio/stop_clip.h"
#include "audio/voice_non_stop_clip.h"
#include "audio/voice_stop_clip.h"
int main(void) {
  stdio_init_all(); // enable USB stdio
  sleep_ms(2000);   // wait for USB serial to come up

  //CLIP1
  /*static float signal1[STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < STOP_CLIP_NUM_SAMPLES; ++i) {
    STOP_CLIP[i] = STOP_CLIP[i] / 32768.0f;
  }*/

  classifier_output_t class = classify_signal(STOP_CLIP, STOP_CLIP_NUM_SAMPLES);
  classifier_binary_output_t binary_class = {
    .stop = (class.stop_prob > 0.5f), 
    .speaker = (class.speaker_llr > 0.4f)
  };

  printf("Stop clip, wrong voice\r\n");
  printf("Probability of 'stop': %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  //CLIP2
  /*static float signal2[NON_STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < NON_STOP_CLIP_NUM_SAMPLES; ++i) {
    NON_STOP_CLIP[i] = NON_STOP_CLIP[i] / 32768.0f;
  }*/

  class = classify_signal(NON_STOP_CLIP, NON_STOP_CLIP_NUM_SAMPLES);
  binary_class.stop = (class.stop_prob > 0.5f);
  binary_class.speaker = (class.speaker_llr > 0.4f);

  printf("Not stop clip, wrong voice\r\n");
  printf("Probability of 'stop': %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  //CLIP3
  /*static float signal3[VOICE_STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < VOICE_STOP_CLIP_NUM_SAMPLES; ++i) {
    VOICE_STOP_CLIP[i] = VOICE_STOP_CLIP[i] / 32768.0f;
  }*/

  class = classify_signal(VOICE_STOP_CLIP, VOICE_STOP_CLIP_NUM_SAMPLES);
  binary_class.stop = (class.stop_prob > 0.5f);
  binary_class.speaker = (class.speaker_llr > 0.4f);

  printf("Stop clip, right voice\r\n");
  printf("Probability of 'stop': %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  //CLIP4
  /*static float signal4[VOICE_NON_STOP_CLIP_NUM_SAMPLES];
  for (int i = 0; i < VOICE_NON_STOP_CLIP_NUM_SAMPLES; ++i) {
    VOICE_NON_STOP_CLIP[i] = VOICE_NON_STOP_CLIP[i] / 32768.0f;
  }*/

  class = classify_signal(VOICE_NON_STOP_CLIP, VOICE_NON_STOP_CLIP_NUM_SAMPLES);
  binary_class.stop = (class.stop_prob > 0.5f);
  binary_class.speaker = (class.speaker_llr > 0.4f);

  printf("Not stop clip, right voice\r\n");
  printf("Probability of 'stop': %f\r\n", class.stop_prob);
  printf("Detected: %d\r\n", binary_class.stop);
  printf("Speaker LLR: %f\r\n", class.speaker_llr);
  printf("Speaker match: %d\r\n", binary_class.speaker);

  // Keep the core alive so USB doesn’t shut down
  while (true) {
    tight_loop_contents();
  }
}
