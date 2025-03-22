#include "ringbuffer.h"

IntRingBuffer *create_int_ring(uint8_t capacity)
{
    IntRingBuffer *buffer = (IntRingBuffer *)malloc(sizeof(IntRingBuffer));

    buffer->ring_buffer = (int16_t *)malloc(capacity * sizeof(int16_t));

    buffer->write = 0;
    buffer->read = 0;
    buffer->size = 0;
    buffer->capacity = capacity;
    return buffer;
}

void ring_buffer_put(IntRingBuffer *buffer, int16_t value)
{
    if (buffer->size == buffer->capacity)
    {
        // if buffer is full, overwrite oldest value
        buffer->read = (buffer->read + 1) % buffer->capacity;
    }
    else
    {
        // increase size if not full
        buffer->size++;
    }

    buffer->ring_buffer[buffer->write] = value;
    buffer->write = (buffer->write + 1) % buffer->capacity;
}


int16_t ring_buffer_peek(IntRingBuffer *buffer)
{
    if (buffer->size == 0)
    {
        // TODO: FIX CAUSE IMU CAN GIVE -1 AS VALUE, USE DIFF. VALUE FOR ERROR
        return -1;  // Return an error value instead of exiting
    }

    int16_t value = buffer->ring_buffer[buffer->read];
    return value;
}


int16_t ring_buffer_get(IntRingBuffer *buffer)
{
    int16_t value = ring_buffer_peek(buffer);
    
    if (value != -1) {
        buffer->read = (buffer->read + 1) % buffer->capacity;
        buffer->size--;
    }

    return value;
}

// Free allocated memory
void free_ring_buffer(IntRingBuffer *buffer)
{
    free(buffer->ring_buffer);
    buffer->ring_buffer = NULL;
}