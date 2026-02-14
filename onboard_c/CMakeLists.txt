cmake_minimum_required(VERSION 3.16)
project(vps_onboard C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -fsanitize=address,undefined")

# ARM NEON optimization (auto-detected on aarch64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-march=armv8-a+simd)
    message(STATUS "ARM64 detected — enabling NEON SIMD")
endif()

# Warning flags
add_compile_options(-Wall -Wextra -Wpedantic -Werror=implicit-function-declaration)

# --- Core library (no external deps) ---
add_library(vps_core STATIC
    src/tile_math.c
    src/geo_transform.c
    src/nmea.c
    src/msp.c
    src/ekf.c
    src/dead_reckoning.c
    src/fusion.c
    src/geofence.c
    src/rate_limiter.c
    src/uart.c
    src/flight_recorder.c
    src/confidence.c
)
target_include_directories(vps_core PUBLIC include)
target_link_libraries(vps_core m)  # libm for math functions

# --- Main executable ---
# add_executable(vps_onboard src/main.c)
# target_link_libraries(vps_onboard vps_core)
# Uncomment when OpenCV and ONNX Runtime are available:
# find_package(OpenCV REQUIRED)
# target_link_libraries(vps_onboard ${OpenCV_LIBS})

# --- Tests ---
enable_testing()

add_executable(test_tile_math tests/test_tile_math.c)
target_link_libraries(test_tile_math vps_core)
add_test(NAME test_tile_math COMMAND test_tile_math)

add_executable(test_nmea tests/test_nmea.c)
target_link_libraries(test_nmea vps_core)
add_test(NAME test_nmea COMMAND test_nmea)

add_executable(test_msp tests/test_msp.c)
target_link_libraries(test_msp vps_core)
add_test(NAME test_msp COMMAND test_msp)

add_executable(test_ekf tests/test_ekf.c)
target_link_libraries(test_ekf vps_core)
add_test(NAME test_ekf COMMAND test_ekf)

add_executable(test_geo_transform tests/test_geo_transform.c)
target_link_libraries(test_geo_transform vps_core)
add_test(NAME test_geo_transform COMMAND test_geo_transform)

add_executable(test_geofence tests/test_geofence.c)
target_link_libraries(test_geofence vps_core)
add_test(NAME test_geofence COMMAND test_geofence)

add_executable(test_fusion tests/test_fusion.c)
target_link_libraries(test_fusion vps_core)
add_test(NAME test_fusion COMMAND test_fusion)

add_executable(test_flight_recorder tests/test_flight_recorder.c)
target_link_libraries(test_flight_recorder vps_core)
add_test(NAME test_flight_recorder COMMAND test_flight_recorder)

add_executable(test_confidence tests/test_confidence.c)
target_link_libraries(test_confidence vps_core)
add_test(NAME test_confidence COMMAND test_confidence)
