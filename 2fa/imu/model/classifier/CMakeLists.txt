cmake_minimum_required(VERSION 3.13)

project(ei_classifier C CXX)

include(edge-impulse-sdk/cmake/utils.cmake)

set(EI_SOURCES "")

# Pull in ALL Edge Impulse SDK sources
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "edge-impulse-sdk" "*.c")
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "edge-impulse-sdk" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "edge-impulse-sdk" "*.cc")

# Pull in compiled model sources
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "tflite-model" "*.c")
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "tflite-model" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(EI_SOURCES "tflite-model" "*.cc")

# Pico-specific porting layer 
list(APPEND EI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ei_pico_porting.cpp
)

add_library(ei_classifier STATIC ${EI_SOURCES})

target_include_directories(ei_classifier PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/edge-impulse-sdk
    ${CMAKE_CURRENT_LIST_DIR}/model-parameters
    ${CMAKE_CURRENT_LIST_DIR}/tflite-model
)

target_compile_features(ei_classifier PUBLIC cxx_std_17)
target_link_libraries(ei_classifier PUBLIC pico_stdlib)
