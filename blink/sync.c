/**
 * Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "pico/stdlib.h"
#include "stdio.h"
#include "hardware/uart.h"

#include "hardware/i2c.h"
#include "mpu6050.h"
#include "ringbuffer.h"
#include "pico/multicore.h"


// Pico W devices use a GPIO on the WIFI chip for the LED,
// so when building for Pico W, CYW43_WL_GPIO_LED_PIN will be defined
#ifdef CYW43_WL_GPIO_LED_PIN
#include "pico/cyw43_arch.h"
#endif

#ifndef LED_DELAY_MS
#define LED_DELAY_MS 250
#endif

#define BAUD_RATE 115200

#define UART_ID uart1


#define UART_TX 8
#define UART_RX 9

// Create 3 ring buffers
IntRingBuffer *ax;
IntRingBuffer *ay;
IntRingBuffer *az;
IntRingBuffer *classification_buffer;

volatile uint8_t print_imu = 0;



// Perform initialisation
int pico_led_init(void) {
#if defined(PICO_DEFAULT_LED_PIN)
    // A device like Pico that uses a GPIO for the LED will define PICO_DEFAULT_LED_PIN
    // so we can use normal GPIO functionality to turn the led on and off
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    return PICO_OK;
#elif defined(CYW43_WL_GPIO_LED_PIN)
    // For Pico W devices we need to initialise the driver etc
    return cyw43_arch_init();
#endif
}

// Turn the led on or off
void pico_set_led(bool led_on) {
#if defined(PICO_DEFAULT_LED_PIN)
    // Just set the GPIO on or off
    gpio_put(PICO_DEFAULT_LED_PIN, led_on);
#elif defined(CYW43_WL_GPIO_LED_PIN)
    // Ask the wifi "driver" to set the GPIO on or off
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_on);
#endif
}

// Spinlock
int spinlock_num_count;
spin_lock_t *spinlock_classification;


void test_ringbuffer() {
    printf("Test 1: Initialization\n");
    IntRingBuffer *buffer = create_int_ring(3);
    printf("Buffer capacity: %d\n", buffer->capacity);


    printf("Test 2: Adding Elements\n");
    ring_buffer_put(buffer, 10);
    ring_buffer_put(buffer, 20);
    ring_buffer_put(buffer, 30);
    
    printf("Test 3: Retrieving Elements\n");
    printf("Expected: 10, Got: %d\n", ring_buffer_get(buffer));
    printf("Expected: 20, Got: %d\n", ring_buffer_get(buffer));


    printf("Test 4: Buffer Wraparound\n");
    ring_buffer_put(buffer, 40);
    ring_buffer_put(buffer, 50);
    

    printf("Expected: 30, Got: %d\n", ring_buffer_get(buffer));


    printf("Expected: 40, Got: %d\n", ring_buffer_get(buffer));

    printf("Test 5: Overwrite Old Data\n");
    ring_buffer_put(buffer, 60);
    ring_buffer_put(buffer, 70);
    ring_buffer_put(buffer, 80); // Should overwrite 50

    printf("Expected: 60, Got: %d\n", ring_buffer_get(buffer));

    printf("Expected: 70, Got: %d\n", ring_buffer_get(buffer));

    printf("Expected: 80, Got: %d\n", ring_buffer_get(buffer));

    printf("Tests Completed.\n");
    free_ring_buffer(buffer); // Cleanup allocated memory
}

void print_imu_buffers(IntRingBuffer* ax, IntRingBuffer* ay, IntRingBuffer* az, IntRingBuffer* classification_buffer) {
    while (ring_buffer_peek(ax) != -1 || ring_buffer_peek(classification_buffer) != -1) {
        printf("Classification_state: %d, ax: %d, ay: %d, az: %d\n", ring_buffer_get(classification_buffer), ring_buffer_get(ax), ring_buffer_get(ay), ring_buffer_get(az));
    }
}

void core1_task() {
    while(true) {
        while(!uart_is_readable);

        // UART
        spin_lock_unsafe_blocking(spinlock_classification);
        char byte = uart_getc(UART_ID);
        printf("%d\n", (int)byte);
        ring_buffer_put(classification_buffer, (int)byte);

        if (byte == 50){
            print_imu = 1;
        }
        spin_unlock_unsafe(spinlock_classification);

        // Necessary apparently
        sleep_ms(1);
    }
}

int main() {
    stdio_init_all();

    // UART init
    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX, GPIO_FUNC_UART);
    gpio_set_function(UART_RX, GPIO_FUNC_UART);

    // Initialize I2C
    // This example will use I2C0 on the default SDA and SCL pins (4, 5 on a Pico)
    i2c_init(I2C_CHAN, 400 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);
    // Make the I2C pins available to picotool
    //bi_decl(bi_2pins_with_func(PICO_DEFAULT_I2C_SDA_PIN, PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C));

    mpu6050_reset();
 
    int16_t acceleration[3], gyro[3];

    int rc = pico_led_init();
    pico_set_led(true);
    hard_assert(rc == PICO_OK);
    // test_ringbuffer();

    uint32_t start =  time_us_32();

    spinlock_num_count = spin_lock_claim_unused(true);
    spinlock_classification = spin_lock_init(spinlock_num_count);

    multicore_launch_core1(core1_task);

    // Create 3 ring buffers
    ax = create_int_ring(100);
    ay = create_int_ring(100);
    az = create_int_ring(100);
    classification_buffer = create_int_ring(100);
    while (true) {
        mpu6050_read_raw(acceleration, gyro);
        // printf("new iteration\n");
        // Write ring buffer values:
        ring_buffer_put(ax, acceleration[0]);
        ring_buffer_put(ay, acceleration[1]);
        ring_buffer_put(az, acceleration[2]);

        if (print_imu) {
            spin_lock_unsafe_blocking(spinlock_classification);
            print_imu = 0;
            print_imu_buffers(ax,ay,az, classification_buffer);
            spin_unlock_unsafe(spinlock_classification);
            sleep_ms(1);
        }
    }
}
