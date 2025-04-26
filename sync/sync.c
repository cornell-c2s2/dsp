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
 
 #include "hardware/irq.h"
 #include "hardware/clocks.h"
 #include "hardware/pll.h"
 #include "hardware/adc.h"
 
 
 #ifndef LED_DELAY_MS
 #define LED_DELAY_MS 250
 #endif
 
 #define BAUD_RATE 115200
 
 // particle connection
 // #define UART_ID uart1
 
 // #define UART_TX 8
 // #define UART_RX 9
 
 // mic connected to GPIO 27
 #define ALARM_NUM 0
 #define ALARM_IRQ TIMER_IRQ_0
 #define DELAY 62.5 // how oftem (in us) IRQ fires
 
 // ADC input 1 on GPIO 27
 #define ADC_PIN 27 // GPIO 27 / ADC1
 #define ADC_CHAN 1
 
 uint16_t filtered_adc;
 #define ADC_CUTOFF 4 // shift by 4 so 12 bit to 8???
 
 // Create 3 ring buffers, define acc. and gyro. arrays
 IntRingBuffer *ax;
 IntRingBuffer *ay;
 IntRingBuffer *az;
 IntRingBuffer *classification_buffer;
 int16_t acceleration[3], gyro[3];
 
 volatile uint8_t print_imu = 0;
 
 // set up DMA channel
 // int mic_chan = dma_claim_unused_channel(true);
 
 // Perform initialisation
 int pico_led_init(void)
 {
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
 void pico_set_led(bool led_on)
 {
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
 
 void test_ringbuffer()
 {
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
 
 void print_imu_buffers(IntRingBuffer *ax, IntRingBuffer *ay, IntRingBuffer *az, IntRingBuffer *classification_buffer)
 {
     while (ring_buffer_peek(ax) != -1 || ring_buffer_peek(classification_buffer) != -1)
     {
         printf("Classification_state: %d, ax: %d, ay: %d, az: %d\n", ring_buffer_get(classification_buffer), ring_buffer_get(ax), ring_buffer_get(ay), ring_buffer_get(az));
     }
 }


 
 void core1_task()
 {
    while(true){
    }
 
 }
 
 
 static void alarm_irq(void) {
     // do stuff in ISR (take a microphone reading)
 
     // digital low pass the adc reading
     uint16_t adc_val = adc_read();
     filtered_adc = filtered_adc + ((adc_val - filtered_adc) >> ADC_CUTOFF);
     printf("%d,", adc_val);
 
 
     mpu6050_read_raw(acceleration, gyro);
     // printf("new iteration\n");
     // Write ring buffer values:
    //  ring_buffer_put(ax, acceleration[0]);
    //  ring_buffer_put(ay, acceleration[1]);
    //  ring_buffer_put(az, acceleration[2]);
 
     if (print_imu)
     {
         spin_lock_unsafe_blocking(spinlock_classification);
         print_imu = 0;
         print_imu_buffers(ax, ay, az, classification_buffer);
         spin_unlock_unsafe(spinlock_classification);
         sleep_ms(1);
     }
     
 
     // clear the alarm irq
     hw_clear_bits(&timer_hw->intr, 1u << ALARM_NUM);
 
     // reset the alarm register
     timer_hw->alarm[ALARM_NUM] = timer_hw->timerawl + DELAY;
 }
 
 int main()
 {
     stdio_init_all();
 
     // Initialize I2C
     // Uses I2C0 on the default SDA and SCL pins (4, 5 on a Pico)
     i2c_init(I2C_CHAN, 400 * 1000);
     gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
     gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
     gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
     gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);
     // Make the I2C pins available to picotool
     // bi_decl(bi_2pins_with_func(PICO_DEFAULT_I2C_SDA_PIN, PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C));
 
     adc_gpio_init(ADC_PIN);
     adc_init();
     adc_select_input(ADC_CHAN); 
     // no divider so adc free-running ????
     // 48 Mhz ADC clock so
     // adc_set_clkdiv(0);
 
     mpu6050_reset();
 
     int rc = pico_led_init();
     pico_set_led(true);
     hard_assert(rc == PICO_OK);
     // test_ringbuffer();
 
     uint32_t start = time_us_32();
 
     spinlock_num_count = spin_lock_claim_unused(true);
     spinlock_classification = spin_lock_init(spinlock_num_count);
 
     multicore_launch_core1(core1_task);
 
     // Create 3 ring buffers
     ax = create_int_ring(100);
     ay = create_int_ring(100);
     az = create_int_ring(100);
     classification_buffer = create_int_ring(100);
 
     // emable the interrupt for the alarm (we're using alarm 0)
     hw_set_bits(&timer_hw->inte, 1u << ALARM_NUM);
     
     // associate an interrupt handler with the alarm_irq
     irq_set_exclusive_handler(ALARM_IRQ, alarm_irq);
 
     // enable the alarm interrupt
     irq_set_enabled(ALARM_IRQ, true);
 
     // write the lower 32 bits of the target time to the alarm register, arming it
     timer_hw->alarm[ALARM_NUM] = timer_hw->timerawl + DELAY;
 
     // nothing happening here
     while(1){}
     return 0;
 }
