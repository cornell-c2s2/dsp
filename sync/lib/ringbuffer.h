// lib/ringbuffer.h
#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t *ring_buffer;
    uint8_t write;
    uint8_t read;
    uint8_t size;
    uint8_t capacity;
} IntRingBuffer;

IntRingBuffer *create_int_ring(uint16_t capacity);

void ring_buffer_put(IntRingBuffer *buffer, int16_t value);

int16_t ring_buffer_peek(IntRingBuffer *buffer);

int16_t ring_buffer_get(IntRingBuffer *buffer);

void free_ring_buffer(IntRingBuffer *buffer);

void ring_buffer_modify_latest(IntRingBuffer *buffer, int16_t new_value);

#ifdef __cplusplus
}
#endif

#endif // RINGBUFFER_H