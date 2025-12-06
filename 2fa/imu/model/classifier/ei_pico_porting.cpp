#include "pico/stdlib.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

extern "C" {
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
}

// Fallback definition if the SDK doesn't provide it
#ifndef EI_WEAK_FN
#define EI_WEAK_FN __attribute__((weak))
#endif

extern "C" {

// ---- Memory functions ----
void *ei_malloc(size_t size) {
    return malloc(size);
}

void *ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

void ei_free(void *ptr) {
    free(ptr);
}

// ---- Timing ----
uint64_t ei_read_timer_us() {
    return time_us_64();
}

uint64_t ei_read_timer_ms() {
    return ei_read_timer_us() / 1000ULL;
}

// ---- Sleep ----
// Header: EI_IMPULSE_ERROR ei_sleep(int32_t time_ms);
EI_IMPULSE_ERROR ei_sleep(int32_t time_ms) {
    if (time_ms < 0) {
        time_ms = 0;
    }
    sleep_ms(static_cast<uint32_t>(time_ms));
    return EI_IMPULSE_OK;   // "no error, continue"
}

// ---- Cancellation hook ----
// Header: EI_IMPULSE_ERROR ei_run_impulse_check_canceled();
EI_IMPULSE_ERROR ei_run_impulse_check_canceled() {
    // Always allow the classifier to keep running
    return EI_IMPULSE_OK;
}

// ---- Print / logging ----
void ei_putchar(char c) {
    // Raw putchar avoids extra translations
    putchar_raw(c);
}

void ei_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

void ei_printf_float(float f) {
    printf("%f", f);
}

// ---- string writer ----
void ei_write_string(char *data, size_t length) {
    // Simple implementation: just print via ei_putchar
    for (size_t i = 0; i < length; ++i) {
        ei_putchar(data[i]);
    }
}

// ---- Device info ----
void ei_device_id(char *device_id, size_t max_length) {
    const char *id = "pico-rp2040";
    snprintf(device_id, max_length, "%s", id);
}

// radio power hook – default to 0 dBm
EI_WEAK_FN void ei_get_tx_power_dbm(int8_t *tx_power_dbm) {
    *tx_power_dbm = 0;
}

} // extern "C"