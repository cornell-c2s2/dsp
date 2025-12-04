#ifndef AUDIO_CLASSIFIER_INFERENCE_H
#define AUDIO_CLASSIFIER_INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns probability [0,1] that the clip contains "stop"
float audio_classifier_predict(const float *mfcc_in_flat);

// Returns 1 if prob > threshold, else 0
int audio_classifier_is_stop(const float *mfcc_in_flat, float threshold);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_CLASSIFIER_INFERENCE_H
