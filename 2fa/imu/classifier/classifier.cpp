#include "pico/stdlib.h"
#include "stdio.h"
#include "hardware/i2c.h"
#include "lib/mpu6050.h"
#include "tflite_learn_845387_6_compiled.h"  // tflite model
#include "trained_model_ops_define.h"       
#include <math.h>
#include <fftw3.h>

#define FEAT_DIM 6
#define SAMPLE_RATE 50 
#define WINDOW_SIZE 256  // Window size 
#define STOP_THRESHOLD 0.05  // Threshold for movement stop detection
#define STOP_WINDOW 50  // Number of iterations to check for stability (movement stop)

// Function to process IMU data (accelerometer/gyroscope) and flatten it for model input
void process_imu_data(int16_t *accel, int16_t *gyro, float *processed_data) {
    for (int i = 0; i < 3; i++) {
        processed_data[i] = accel[i] / 32768.0f;  // Normalize 
        processed_data[i + 3] = gyro[i] / 32768.0f; 
    }
}

// Function to perform FFT (Spectral Analysis) on  IMU data
void perform_fft(double *data, int data_size, double *frequencies, double *spectrum) {
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * data_size);
    fftw_plan p = fftw_plan_dft_r2c_1d(data_size, data, out, FFTW_ESTIMATE);
    
    fftw_execute(p); 

    // Extract frequencies and spectrum magnitudes
    for (int i = 0; i < data_size / 2 + 1; i++) {
        frequencies[i] = (double)i * SAMPLE_RATE / data_size;
        spectrum[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }
    
    fftw_destroy_plan(p);
    fftw_free(out);
}

// classification using tensorflow lite model
int classify_imu_data(float *processed_data) {
    // Perform the inference
    float prediction = audio_classifier_predict(processed_data);

    printf("Prediction: %f\n", prediction);

    // Assuming a threshold of 0.5 for binary classification (e.g., "motion detected" or "no motion")
    if (prediction > 0.5) {
        printf("Classified as: Motion Detected\n");
        return 1; 
    } else {
        printf("Classified as: No Motion\n");
        return 0;  
    }
}
