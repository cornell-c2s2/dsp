#include "pico/stdlib.h"
#include "stdio.h"
#include "hardware/uart.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/adc.h"
#include "pico/multicore.h"
#include "hardware/dma.h"
#include <string.h>
#include <fstream>
#include <iostream>

#include "lib/mpu6050.h"
#include "lib/ringbuffer.h"
#include "lib/classifier.h"

#ifndef LED_DELAY_MS
#define LED_DELAY_MS 250
#endif

#define UART_ID uart0
#define UART_TX_PIN 0
#define UART_RX_PIN 1

#define BAUD_RATE 115200
#define ALARM_NUM 0
#define ALARM_IRQ TIMER_IRQ_0
#define IMU_IRQ TIMER_IRQ_1

// how often (in us) IRQ fires
#define IMU_DELAY 20000
#define ADC_DELAY 62

#define ADC_PIN 27 // GPIO 27 / ADC1
#define ADC_CHAN 1

uint16_t filtered_adc;
#define ADC_CUTOFF 2 // shift by 4 bits

#define DMA_UART_CHANNEL 0

// Ring buffers:

#define FRAME_SIZE 50
#define WINDOW_SIZE 4

int frame_index = 0;
int frame_count = 0;

// Struct containing 1 sec of information
typedef struct {
    int16_t ax[FRAME_SIZE];
    int16_t ay[FRAME_SIZE];
    int16_t az[FRAME_SIZE];

    int16_t gx[FRAME_SIZE];
    int16_t gy[FRAME_SIZE];
    int16_t gz[FRAME_SIZE];

    // 0 = no class, 1 = no bird, 2 = scrub jay
    int8_t classification;  
} IMUClassificationFrame;

// 4 second imu window
IMUClassificationFrame imu_window[WINDOW_SIZE];


int16_t acceleration[3], gyro[3], temp;
volatile uint8_t print_imu = 0;

// Spinlock
int spinlock_num_count;


static unsigned long lastSampleTime = 0; // Time of last sample
const int sampleRate = 16000;            // Actually closer to 9000 (hardware limitations?)
int count = 0;                           // Number of samples in buffer
const int BUF_SIZE = 16000;              // Size of the buffer
static float buffer[BUF_SIZE];           // Collect samples for the classifier
bool sample_from_adc = false; // Hit noise requirement
bool use_classifier = false;

// Core1 task: run classifier when flag is true (from alarm IRQ)
void core1_task()
{
    // while (true)
    // {
    //     if (use_classifier)
    //     {

    //         //for(int i =0 ; i<BUF_SIZE;i++){printf("%.6f,",buffer[i]);}
    //         print_imu = classify(buffer, (sizeof(buffer) / sizeof(buffer[0]))) + 1;
    //         imu_window[frame_index].classification = print_imu;
    //         // reset to start listening again
    //         count = 0;
    //         sample_from_adc = false;
    //         use_classifier = false;
    //         printf("\nWaiting for noise...\n");
    //     }
        
    //     sleep_ms(1);
    // }
}

bool IMU_callback(struct repeating_timer *t)
{
    static bool imu_pin_state = false;
    gpio_put(12, imu_pin_state);
    imu_pin_state = !imu_pin_state;

    // Get new accelerometer values
    mpu6050_read_raw(acceleration, gyro, &temp);

    // Store into current frame
    if (frame_count < FRAME_SIZE) {
        imu_window[frame_index].ax[frame_count] = acceleration[0];
        imu_window[frame_index].ay[frame_count] = acceleration[1];
        imu_window[frame_index].az[frame_count] = acceleration[2];

        imu_window[frame_index].gx[frame_count] = gyro[0];
        imu_window[frame_index].gy[frame_count] = gyro[1];
        imu_window[frame_index].gz[frame_count] = gyro[2];

        frame_count++;
    }

    // When 1 second of samples is collected
    if (frame_count >= FRAME_SIZE) {
        // default/unclassified
        imu_window[frame_index].classification = 0; 
        frame_index = (frame_index + 1) % WINDOW_SIZE;  // move to next second
        frame_count = 0;
    }
    return true;


}


// Main function
int main()
{
    stdio_init_all();

    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);

    // // Initialize I2C
    i2c_init(MPU6050_I2C, 400 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);

    // adc_gpio_init(ADC_PIN);
    // adc_init();
    // adc_select_input(ADC_CHAN);

    mpu6050_reset();
    // For checking if timing for ADC is met
    gpio_init(13);
    gpio_set_dir(13, GPIO_OUT);
    gpio_put(13, 0);

    // For checking if timer for IMU is met
    gpio_init(12);
    gpio_set_dir(12, GPIO_OUT);
    gpio_put(12, 0);

    spinlock_num_count = spin_lock_claim_unused(true);

    // Put in core 1
    //multicore_launch_core1(core1_task);
    alarm_pool_t *core0pool = alarm_pool_create(0, 2);
    struct repeating_timer imu_sample;
    alarm_pool_add_repeating_timer_us(core0pool, -IMU_DELAY, IMU_callback, NULL, &imu_sample);

    // std::ofstream outputFile("newimudata.csv");
    // if (!outputFile.is_open()) {
    //     printf("Error opening file!");
    //     return 1;
    // }

    // outputFile << "x,y,z,ax,ay,az" << std::endl;

    int samples = 0;
    while(samples < 5000) {
        printf("%d, %d, %d, %d, %d, %d, %d\n", acceleration[0], acceleration[1], acceleration[2], gyro[0], gyro[1], gyro[2], temp);
        samples++;

        //outputFile << acceleration[0] << "," << acceleration[1] << "," << acceleration[2] << "," << gyro[0] << "," << gyro[1] << "," << gyro[2] << std::endl;


    };
    //outputFile.close();
    return 0;
}