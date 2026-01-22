cmake_minimum_required(VERSION 3.16)

project(su3_alg
    VERSION 1.0
    LANGUAGES C
)
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Compiler Version: ${CMAKE_C_COMPILER_VERSION}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(su3lib
    ${PROJECT_SOURCE_DIR}/modules/ufields.c
    ${PROJECT_SOURCE_DIR}/modules/su3.c
    ${PROJECT_SOURCE_DIR}/modules/su3v.c
)

target_include_directories(su3lib
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(su3lib PUBLIC OpenMP::OpenMP_C)
        target_compile_definitions(su3lib PUBLIC SU3_USE_OPENMP)
    else()
        message(WARNING "OpenMP requested but not found; building without OpenMP")
    endif()
endif()

target_compile_options(su3lib PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O3 -march=native
)
#  AVX-512
target_compile_options(su3lib PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl -fopt-info-vec-missed=vec_missed.txt)
add_subdirectory(main)
