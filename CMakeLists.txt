cmake_minimum_required(VERSION 3.16)

project(su3_alg
    VERSION 1.0
    LANGUAGES C
)

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


option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Offloading" OFF)
if(ENABLE_GPU_OFFLOAD)
    target_compile_definitions(su3lib PUBLIC SU3_USE_GPU_OFFLOAD)

    target_compile_options(su3lib PUBLIC -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        --offload-arch=sm_86
    )
    target_link_options(su3lib PUBLIC -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        --offload-arch=sm_86
    )

    message(STATUS "GPU Offloading Enabled")
    message(STATUS "AVX disabled when using GPU")
    set(ENABLE_AVX OFF)
else()
    message(STATUS "GPU Offloading NOT Enabled")
endif()


option(ENABLE_AVX "Enable AVX vectorization" OFF)
if(ENABLE_AVX)
    #  AVX-512
    target_compile_options(su3lib PRIVATE -mavx512f -mavx512dq -mavx512bw
        -mavx512vl)
    message(STATUS "AVX enabled")
else()
    # No autovectorization
    target_compile_options(su3lib PRIVATE
        -fno-tree-vectorize
        -fno-tree-slp-vectorize
    )
    message(STATUS "AVX NOT enabled")
endif()

target_compile_options(su3lib PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O
    -std=c11
)


add_subdirectory(main)
