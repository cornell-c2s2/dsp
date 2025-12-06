# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.1)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.1.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PICO_BOARD pico CACHE STRING "Board type")

include(pico_sdk_import.cmake)

#project(main C CXX ASM) # originally project(sync C CXX ASM)
project(sync C CXX ASM)
pico_sdk_init()

# Add your libraries folders
add_subdirectory(lib)
add_subdirectory(classifier)

add_executable(sync
    sync.cpp
)

target_include_directories(sync PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/lib
    ${CMAKE_CURRENT_LIST_DIR}/classifier
)

# Link pico libraries + your own
target_link_libraries(sync
    pico_stdlib
    hardware_uart
    hardware_i2c
    hardware_adc
    hardware_irq
    hardware_clocks
    hardware_pll
    hardware_dma
    pico_multicore
    mpu6050
    ringbuffer
    ei_classifier
)

if (PICO_CYW43_SUPPORTED)
    target_link_libraries(sync pico_cyw43_arch_none)
endif()

pico_add_extra_outputs(sync)