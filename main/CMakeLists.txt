
add_executable(main main.c)
target_include_directories(main
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main PRIVATE su3lib)
target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic)