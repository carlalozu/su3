function(configure_exec target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(${target} PRIVATE su3lib)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

add_executable(main main.c)
configure_exec(main)

add_executable(main_fields main_fields.c)
configure_exec(main_fields)
