 #include "pico/stdlib.h"
 #include "stdio.h"
 #include "hardware/uart.h"
 #include "hardware/i2c.h"
 #include "hardware/irq.h"
 #include "hardware/clocks.h"
 #include "hardware/pll.h"
 #include "hardware/adc.h"
 #include "pico/multicore.h"
 
 #include "lib/mpu6050.h"
 #include "lib/ringbuffer.h"
 #include "lib/classifier.h" 
 
 #ifndef LED_DELAY_MS
 #define LED_DELAY_MS 250
 #endif
 
 #define BAUD_RATE 115200
 #define ALARM_NUM 0
 #define ALARM_IRQ TIMER_IRQ_0
 #define DELAY 62.5 // how often (in us) IRQ fires
 
 #define ADC_PIN 27 // GPIO 27 / ADC1
 #define ADC_CHAN 1
 
 uint16_t filtered_adc;
 #define ADC_CUTOFF 4 // shift by 4 bits
 
 // Ring buffers
 IntRingBuffer *ax;
 IntRingBuffer *ay;
 IntRingBuffer *az;
 IntRingBuffer *classification_buffer;
 
 int16_t acceleration[3], gyro[3];
 volatile uint8_t print_imu = 0;
 
 // Spinlock
 int spinlock_num_count;
 spin_lock_t *spinlock_classification;
 
 // Ring buffer testing
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
     ring_buffer_put(buffer, 80);
 
     printf("Expected: 60, Got: %d\n", ring_buffer_get(buffer));
     printf("Expected: 70, Got: %d\n", ring_buffer_get(buffer));
     printf("Expected: 80, Got: %d\n", ring_buffer_get(buffer));
 
     printf("Tests Completed.\n");
     free_ring_buffer(buffer);
 }
 
 // Print ring buffers
 void print_imu_buffers(IntRingBuffer *ax, IntRingBuffer *ay, IntRingBuffer *az, IntRingBuffer *classification_buffer) {
     while (ring_buffer_peek(ax) != -1 || ring_buffer_peek(classification_buffer) != -1) {
         printf("Classification_state: %d, ax: %d, ay: %d, az: %d\n",
             ring_buffer_get(classification_buffer),
             ring_buffer_get(ax),
             ring_buffer_get(ay),
             ring_buffer_get(az));
     }
 }
 
 // LED initialization
 int pico_led_init(void) {
 #if defined(PICO_DEFAULT_LED_PIN)
     gpio_init(PICO_DEFAULT_LED_PIN);
     gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
     return PICO_OK;
 #elif defined(CYW43_WL_GPIO_LED_PIN)
     return cyw43_arch_init();
 #endif
 }
 
 // LED control
 void pico_set_led(bool led_on) {
 #if defined(PICO_DEFAULT_LED_PIN)
     gpio_put(PICO_DEFAULT_LED_PIN, led_on);
 #elif defined(CYW43_WL_GPIO_LED_PIN)
     cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_on);
 #endif
 }


static unsigned long lastSampleTime = 0;  // Time of last sample
const int sampleRate = 16000;             // Actually closer to 9000 (hardware limitations?)
int count = 0;                            // Number of samples in buffer
const int BUF_SIZE = 3807;                // Size of the buffer
static float buffer[BUF_SIZE];            // Collect samples for the classifier
const int UPBUF_SIZE = BUF_SIZE * 16 / 9; // 9000 Hz to 16000 Hz
static float upsampledBuffer[UPBUF_SIZE]; // Buffer after upsampling
bool flag = false;                        // Hit noise requirement

 
 // Core1 task: run classifier when flag is true (from alarm IRQ)
 void core1_task() {
     while (true) {
        if (!flag) {
            printf("Waiting for noise...\n");
            //bruh
        }
        else {
            printf("Begin Classification...\n");
            classify(upsampledBuffer, (sizeof(upsampledBuffer) / sizeof(upsampledBuffer[0])));
        }
     }
 }
 
 // Alarm interrupt handler (mic reading)
 static void alarm_irq(void) {
     uint16_t adc_val = adc_read();
     filtered_adc = filtered_adc + ((adc_val - filtered_adc) >> ADC_CUTOFF);
    //  printf("%d,", adc_val);

    if (filtered_adc > 1000)
    {
      flag = true;
    }
    if (flag)
    {
      // Convert 12-bit ADC value to signed 16-bit PCM
      int16_t pcmSample = (adc_val - 2048) * 16;

      if (count < BUF_SIZE) // Collect data
      {
        buffer[count++] = pcmSample;
      }
      else if (count == BUF_SIZE) // Data collected
      {
        // Normalize the data to [-1, 1]
        for (int i = 0; i < UPBUF_SIZE; i++)
        {
          upsampledBuffer[i] = upsampledBuffer[i] / 32768.0;
        }
      }
    }

     //mpu logic
     mpu6050_read_raw(acceleration, gyro);
 
     if (print_imu) {
         spin_lock_unsafe_blocking(spinlock_classification);
         print_imu = 0;
         print_imu_buffers(ax, ay, az, classification_buffer);
         spin_unlock_unsafe(spinlock_classification);
         sleep_ms(1);
     }
 
     hw_clear_bits(&timer_hw->intr, 1u << ALARM_NUM);
     timer_hw->alarm[ALARM_NUM] = timer_hw->timerawl + DELAY;
 }
 
 // Main function
 int main() {
     stdio_init_all();
 
     // Initialize I2C
     i2c_init(MPU6050_I2C, 400 * 1000);
     gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
     gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
     gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
     gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);
 
     adc_gpio_init(ADC_PIN);
     adc_init();
     adc_select_input(ADC_CHAN);
 
     mpu6050_reset();
 
     int rc = pico_led_init();
     pico_set_led(true);
     hard_assert(rc == PICO_OK);
 
     spinlock_num_count = spin_lock_claim_unused(true);
     spinlock_classification = spin_lock_init(spinlock_num_count);
 
     multicore_launch_core1(core1_task);
 
     ax = create_int_ring(100);
     ay = create_int_ring(100);
     az = create_int_ring(100);
     classification_buffer = create_int_ring(100);
 
     hw_set_bits(&timer_hw->inte, 1u << ALARM_NUM);
     irq_set_exclusive_handler(ALARM_IRQ, alarm_irq);
     irq_set_enabled(ALARM_IRQ, true);
     timer_hw->alarm[ALARM_NUM] = timer_hw->timerawl + DELAY;
 
     while (true) {
         // Main loop
     }
 
     return 0;
 }