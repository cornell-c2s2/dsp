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

// accelerometer
IntRingBuffer *ax;
IntRingBuffer *ay;
IntRingBuffer *az;

// gyro
IntRingBuffer *gx;
IntRingBuffer *gy;
IntRingBuffer *gz;


IntRingBuffer *classification_buffer;

int16_t acceleration[3], gyro[3], temp;
volatile uint8_t print_imu = 0;

// Spinlock
int spinlock_num_count;
spin_lock_t *spinlock_classification;

void dma_uart_print(const char *str) {
    int len = strlen(str);
    dma_channel_config cfg = dma_channel_get_default_config(DMA_UART_CHANNEL);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_8);
    channel_config_set_read_increment(&cfg, true);
    channel_config_set_write_increment(&cfg, false);
    channel_config_set_dreq(&cfg, uart_get_dreq(UART_ID, true));

    dma_channel_configure(
        DMA_UART_CHANNEL,
        &cfg,
        &uart_get_hw(UART_ID)->dr,  // UART FIFO
        str,
        len,
        true  // Start immediately
    );
    dma_channel_wait_for_finish_blocking(DMA_UART_CHANNEL);
}


// Ring buffer testing
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
    ring_buffer_put(buffer, 80);

    printf("Expected: 60, Got: %d\n", ring_buffer_get(buffer));
    printf("Expected: 70, Got: %d\n", ring_buffer_get(buffer));
    printf("Expected: 80, Got: %d\n", ring_buffer_get(buffer));

    printf("Tests Completed.\n");
    free_ring_buffer(buffer);
}

// Print ring buffers
void print_imu_buffers(IntRingBuffer *ax, IntRingBuffer *ay, IntRingBuffer *az, IntRingBuffer *classification_buffer)
{
    char buffer[128];  // temp string buffer
    while (ring_buffer_peek(ax) != -1 || ring_buffer_peek(classification_buffer) != -1)
    {
        int cls = ring_buffer_get(classification_buffer);
        int x = ring_buffer_get(ax);
        int y = ring_buffer_get(ay);
        int z = ring_buffer_get(az);

        int gx_val = ring_buffer_get(gx);
        int gy_val = ring_buffer_get(gy);
        int gz_val = ring_buffer_get(gz);

        snprintf(buffer, sizeof(buffer),
                "Cls: %d, ax: %d, ay: %d, az: %d, gx: %d, gy: %d, gz: %d\n",
                cls, x, y, z, gx_val, gy_val, gz_val);

        dma_uart_print(buffer);
    }
}


// LED initialization
int pico_led_init(void)
{
#if defined(PICO_DEFAULT_LED_PIN)
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    return PICO_OK;
#elif defined(CYW43_WL_GPIO_LED_PIN)
    return cyw43_arch_init();
#endif
}

// LED control
void pico_set_led(bool led_on)
{
#if defined(PICO_DEFAULT_LED_PIN)
    gpio_put(PICO_DEFAULT_LED_PIN, led_on);
#elif defined(CYW43_WL_GPIO_LED_PIN)
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_on);
#endif
}

static unsigned long lastSampleTime = 0; // Time of last sample
const int sampleRate = 16000;            // Actually closer to 9000 (hardware limitations?)
int count = 0;                           // Number of samples in buffer
const int BUF_SIZE = 16000;              // Size of the buffer
static float buffer[BUF_SIZE];           // Collect samples for the classifier
// const int UPBUF_SIZE = BUF_SIZE * 16 / 9; // 9000 Hz to 16000 Hz
// static float upsampledBuffer[UPBUF_SIZE]; // Buffer after upsampling
bool sample_from_adc = false; // Hit noise requirement
bool use_classifier = false;

// Core1 task: run classifier when flag is true (from alarm IRQ)
void core1_task()
{
    while (true)
    {
        if (use_classifier)
        {
            for(int i =0 ; i<BUF_SIZE;i++){printf("%.6f,",buffer[i]);}
            print_imu = classify(buffer, (sizeof(buffer) / sizeof(buffer[0])));
            
            // reset to start listening again
            count = 0;
            sample_from_adc = false;
            use_classifier = false;
            printf("Waiting for noise...\n");
        }

        if (print_imu)
        {
            spin_lock_unsafe_blocking(spinlock_classification);
            print_imu = 0;
            //print_imu_buffers(ax, ay, az, classification_buffer);
            spin_unlock_unsafe(spinlock_classification);
            // SPINLOCK DOCUMENTATION SAYS TO SLEEP FOR 1 MS
            sleep_ms(1);
        }
        
        sleep_ms(1);
    }
}

// Alarm interrupt handler (mic reading)
bool adc_callback(struct repeating_timer *t)
{
    // This code creates a square wave on PIN 17 of the PICO
    // 2 square waves is how long the alarm_irq runs for
    static bool pin_state = false;
    gpio_put(13, pin_state);
    pin_state = !pin_state;

    uint16_t filtered_adc = adc_read();
    // filtered_adc = filtered_adc + ((adc_val - filtered_adc) >> ADC_CUTOFF);
    // printf("%d,", adc_val);

    if ((filtered_adc < 1548 || filtered_adc > 2548) && !sample_from_adc)
    {
        //printf("Start");
        sample_from_adc = true;
    }

    if (sample_from_adc)
    {
        // Convert 12-bit ADC value to signed 16-bit PCM
        int16_t pcmSample = (filtered_adc - 2048) * 16;

        if (count < BUF_SIZE) // Collect data
        {
            // Normalize the data to [-1, 1]
            buffer[count++] = pcmSample / 32768.0;
        }
        else if (!use_classifier)
        {

            //printf("end");

            use_classifier = true;
        }
    }
    ring_buffer_put(classification_buffer, 0);

    return true;
}

bool IMU_callback(struct repeating_timer *t)
{
    static bool imu_pin_state = false;
    gpio_put(12, imu_pin_state);
    imu_pin_state = !imu_pin_state;

    // Get new accelerometer values
    mpu6050_read_raw(acceleration, gyro, &temp);

    ring_buffer_put(ax, acceleration[0]);
    ring_buffer_put(ay, acceleration[1]);
    ring_buffer_put(az, acceleration[2]);

    ring_buffer_put(gx, gyro[0]);
    ring_buffer_put(gy, gyro[1]);
    ring_buffer_put(gz, gyro[2]);


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

    adc_gpio_init(ADC_PIN);
    adc_init();
    adc_select_input(ADC_CHAN);

    mpu6050_reset();
    // For checking if timing for ADC is met
    gpio_init(13);
    gpio_set_dir(13, GPIO_OUT);
    gpio_put(13, 0);

    // For checking if timer for IMU is met
    gpio_init(12);
    gpio_set_dir(12, GPIO_OUT);
    gpio_put(12, 0);

    int rc = pico_led_init();
    pico_set_led(true);
    hard_assert(rc == PICO_OK);

    spinlock_num_count = spin_lock_claim_unused(true);
    spinlock_classification = spin_lock_init(spinlock_num_count);

    // Put in core 1
    multicore_launch_core1(core1_task);

    // acc
    ax = create_int_ring(1000);
    ay = create_int_ring(1000);
    az = create_int_ring(1000);

    // gyro
    gx = create_int_ring(1000);
    gy = create_int_ring(1000);
    gz = create_int_ring(1000);

    classification_buffer = create_int_ring(1000);

    // We will create an alarm pool on core 0, this main() function
    alarm_pool_t *core0pool = alarm_pool_create(0, 2);

    // Initalize structs for alarm pool functions
    struct repeating_timer imu_sample;
    struct repeating_timer adc_sample;

    alarm_pool_add_repeating_timer_us(core0pool, -ADC_DELAY, adc_callback, NULL, &adc_sample);
    alarm_pool_add_repeating_timer_us(core0pool, -IMU_DELAY, IMU_callback, NULL, &imu_sample);

    while(true) {

    };

    return 0;
}