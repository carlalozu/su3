
function(configure_exec target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(${target} PRIVATE su3lib)
    target_compile_options(${target} PRIVATE ${CMAKE_C_FLAGS})
endfunction()

add_executable(su3_alg su3_alg.c)
configure_exec(su3_alg)

add_executable(su3_fields su3_fields.c)
configure_exec(su3_fields)

add_executable(su3_fieldsv su3_fieldsv.c)
configure_exec(su3_fieldsv)

if(ENABLE_GPU_OFFLOAD)
    add_executable(time_gpu time_gpu.c)
    configure_exec(time_gpu)
endif()

add_executable(time_cpu time_cpu.c)
configure_exec(time_cpu)
