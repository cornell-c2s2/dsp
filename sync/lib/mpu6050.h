// lib/mpu6050.h
#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>
#include <stdbool.h>
#include "hardware/i2c.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MPU6050_ADDRESS 0x68
#define MPU6050_I2C     i2c0


void mpu6050_reset(void);
void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp);

#ifdef __cplusplus
}
#endif

#endif // MPU6050_H