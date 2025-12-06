#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>

#include "mfcc_params.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute MFCCs for a mono float signal (16 kHz) and write
// up to max_frames frames of MFCC_N_MFCC coefficients.
// out_mfcc must have size at least max_frames * MFCC_N_MFCC.
// Returns number of frames actually computed.
int compute_mfcc(const float *signal,
                 int num_samples,
                 float *out_mfcc,
                 int max_frames);

#ifdef __cplusplus
}
#endif

#endif // MFCC_H
