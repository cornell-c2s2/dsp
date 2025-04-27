#include "mpu6050.h"
#include "pico/stdlib.h"

void mpu6050_reset() {
    uint8_t buf[] = {0x6B, 0x80}; // Reset device
    i2c_write_blocking(MPU6050_I2C, MPU6050_ADDRESS, buf, 2, false);
    sleep_ms(100); // Allow device to reset and stabilize
}

void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3]) {
    uint8_t buf[14];
    uint8_t reg = 0x3B; // starting register for accelerometer data
    i2c_write_blocking(MPU6050_I2C, MPU6050_ADDRESS, &reg, 1, true);
    i2c_read_blocking(MPU6050_I2C, MPU6050_ADDRESS, buf, 14, false);

    accel[0] = (int16_t)(buf[0] << 8 | buf[1]);
    accel[1] = (int16_t)(buf[2] << 8 | buf[3]);
    accel[2] = (int16_t)(buf[4] << 8 | buf[5]);

    gyro[0] = (int16_t)(buf[8] << 8 | buf[9]);
    gyro[1] = (int16_t)(buf[10] << 8 | buf[11]);
    gyro[2] = (int16_t)(buf[12] << 8 | buf[13]);
}