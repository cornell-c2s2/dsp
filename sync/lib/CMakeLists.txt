add_library(mpu6050 mpu6050.c)
add_library(ringbuffer ringbuffer.c)
 add_library(classifier
     classifier.cpp
     PlainFFT.cpp
 )
# This will correctly add all SDK include paths and defines
target_link_libraries(mpu6050 pico_stdlib hardware_i2c)
target_link_libraries(ringbuffer pico_stdlib)
target_link_libraries(classifier pico_stdlib)

# (Optional) If you want clean headers visibility for your libs
target_include_directories(mpu6050 PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(ringbuffer PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(classifier PUBLIC ${CMAKE_CURRENT_LIST_DIR})