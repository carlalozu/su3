add_executable(su3_alg
    main.c
    su3.c
)

target_include_directories(su3_alg
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
