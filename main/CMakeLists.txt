add_executable(main
    main.c
)

target_include_directories(main
    PRIVATE
        ${PROJECT_SOURCE_DIR}/modules
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic)