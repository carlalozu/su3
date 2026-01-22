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

add_subdirectory(main)
