 // By default these devices are on bus address 0x68
#define ADDRESS 0x68

void mpu6050_reset();

void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3]);