#include "pico/stdlib.h"
#include "stdio.h"
#include "hardware/i2c.h"

#include "lib/mpu6050.h"

// Main function
int main()
{
    // Initialize stdio for USB serial
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

    // Print CSV header
    printf("ax,ay,az,gx,gy,gz,temp\n");

    // Continuously read and print IMU data
    while (true) {
        // Read IMU data
        mpu6050_read_raw(acceleration, gyro, &temp);
        
        // Print in CSV format
        printf("%d,%d,%d,%d,%d,%d,%d\n", 
               acceleration[0], acceleration[1], acceleration[2],
               gyro[0], gyro[1], gyro[2]); 
               //temp);
        
        // Small delay to control sampling rate (~50Hz)
        sleep_ms(20);
    }

    return 0;
}