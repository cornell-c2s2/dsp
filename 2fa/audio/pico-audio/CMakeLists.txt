cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico-audio C CXX ASM)
pico_sdk_init()

add_executable(pico_stop_detector
    src/main.c
    src/mfcc.c
    src/audio_classifier_inference.c
    src/stop_detector.c
    src/speaker_gmm.c
)


target_include_directories(pico_stop_detector PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
)

target_link_libraries(pico_stop_detector
    pico_stdlib
    m 
)

pico_enable_stdio_usb(pico_stop_detector 1)
pico_enable_stdio_uart(pico_stop_detector 0)

pico_add_extra_outputs(pico_stop_detector)