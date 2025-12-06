cmake_minimum_required(VERSION 3.16)
project(scrubjay_infer C)

set(CMAKE_C_STANDARD 11)

# adjust ORT_ROOT if you unpacked the archive somewhere else
set(ORT_ROOT $ENV{ORT_ROOT})

find_package(PkgConfig REQUIRED)
pkg_check_modules(AUBIO REQUIRED aubio)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3 REQUIRED fftw3f)

include_directories(${AUBIO_INCLUDE_DIRS} ${SNDFILE_INCLUDE_DIRS}
                    ${FFTW3_INCLUDE_DIRS} ${ORT_ROOT}/include)

link_directories(${AUBIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS}
                 ${FFTW3_LIBRARY_DIRS} ${ORT_ROOT})

add_executable(scrubjay_infer scrubjay_infer.c)

target_link_libraries(scrubjay_infer
    ${AUBIO_LIBRARIES} ${SNDFILE_LIBRARIES} ${FFTW3_LIBRARIES}
    ${ORT_ROOT}/lib/libonnxruntime.so)          # .dylib on macOS
