cmake_minimum_required(VERSION 3.10)

project(2FAudio C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wconversion -Wshadow -Wsign-conversion")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=signed-integer-overflow -ftrapv")

file(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/*.h")
foreach(header_file ${HEADER_FILES})
    get_filename_component(dir ${header_file} DIRECTORY)
    include_directories(${dir})
endforeach()

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")

add_executable(2FAudio ${SOURCES})

target_link_libraries(2FAudio m)
