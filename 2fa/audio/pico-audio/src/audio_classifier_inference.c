#include "audio_classifier_inference.h"
#include "model_params.h"

#include <math.h>

// Simple ReLU
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Sigmoid
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Dense layer: y = activation(W * x + b)
// W is (in_dim x out_dim) flattened row-major (as in NumPy)
static void dense_forward(const float *x,
                          const float *W,
                          const float *b,
                          int in_dim,
                          int out_dim,
                          float *y,
                          int apply_relu)
{
    // We stored W in NumPy's default layout: (in_dim, out_dim) row-major
    // So W[i*out_dim + j] is the weight from input i to output j.
    for (int j = 0; j < out_dim; ++j) {
        float sum = b[j];
        for (int i = 0; i < in_dim; ++i) {
            sum += W[i * out_dim + j] * x[i];
        }
        y[j] = apply_relu ? relu(sum) : sum;
    }
}

// mfcc_in_flat must have length INPUT_SIZE
float audio_classifier_predict(const float *mfcc_in_flat)
{
    // 1) Standardize input using saved scaler
    static float x_scaled[INPUT_SIZE];  // static to avoid stack blowup
    for (int i = 0; i < INPUT_SIZE; ++i) {
        float scale = SCALER_SCALE[i];
        // Avoid divide-by-zero (shouldn't happen, but just in case)
        if (scale == 0.0f) scale = 1.0f;
        x_scaled[i] = (mfcc_in_flat[i] - SCALER_MEAN[i]) / scale;
    }

    // 2) Hidden layers
    float h1[DENSE1_UNITS];
    float h2[DENSE2_UNITS];
    float h3[DENSE3_UNITS];

    dense_forward(x_scaled,
                  DENSE1_KERNEL,
                  DENSE1_BIAS,
                  INPUT_SIZE,
                  DENSE1_UNITS,
                  h1,
                  1); // ReLU

    dense_forward(h1,
                  DENSE2_KERNEL,
                  DENSE2_BIAS,
                  DENSE1_UNITS,
                  DENSE2_UNITS,
                  h2,
                  1); // ReLU

    dense_forward(h2,
                  DENSE3_KERNEL,
                  DENSE3_BIAS,
                  DENSE2_UNITS,
                  DENSE3_UNITS,
                  h3,
                  1); // ReLU

    float out_raw[DENSE4_UNITS]; // should be size 1
    dense_forward(h3,
                  DENSE4_KERNEL,
                  DENSE4_BIAS,
                  DENSE3_UNITS,
                  DENSE4_UNITS,
                  out_raw,
                  0); // no ReLU

    // 3) Sigmoid output
    float prob = sigmoid(out_raw[0]);
    return prob;
}

int audio_classifier_is_stop(const float *mfcc_in_flat, float threshold)
{
    float p = audio_classifier_predict(mfcc_in_flat);
    return (p > threshold) ? 1 : 0;
}
