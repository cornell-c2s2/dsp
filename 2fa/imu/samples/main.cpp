#include "pico/stdlib.h"
#include "stdio.h"
#include "hardware/i2c.h"
#include "lib/mpu6050.h"
#include "../classifier/classifier.cpp"

#define SAMPLE_RATE 50

int main() {
    stdio_init_all();

    // Initialize I2C for MPU6050
    i2c_init(MPU6050_I2C, 400 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);

    // Reset and initialize MPU6050
    mpu6050_reset();

    // Variables to store IMU data
    int16_t acceleration[3], gyro[3], temp;
    float processed_data[FEAT_DIM];  // Array for normalized IMU data

    // Continuously read and classify IMU data
    while (true) {
        // Read IMU data
        mpu6050_read_raw(acceleration, gyro, &temp);

        // Process the IMU data to prepare it for the model
        process_imu_data(acceleration, gyro, processed_data);

        // Classify the IMU data (motion detection)
        classify_imu_data(processed_data);

        // Small delay to control sampling rate (~50Hz)
        sleep_ms(20);
    }

    return 0;
}
