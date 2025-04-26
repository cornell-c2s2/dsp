add_library(mpu6050 STATIC mpu6050.c)

target_include_directories(mpu6050 PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(mpu6050 
                    pico_stdlib 
                    hardware_i2c
                    hardware_irq
                    hardware_clocks
                    hardware_pll
                    hardware_adc
                    ) 

add_library(ringbuffer STATIC ringbuffer.c)

target_include_directories(ringbuffer PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(ringbuffer pico_stdlib)

add_library(PlainFFT STATIC PlainFFT.cpp)

target_include_directories(PlainFFT PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(PlainFFT)
