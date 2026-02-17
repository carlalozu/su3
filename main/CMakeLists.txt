
macro(add_su3_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE su3lib)
endmacro()

add_su3_executable(su3_alg su3_alg.c)
add_su3_executable(su3_fields su3_fields.c)
add_su3_executable(su3_fieldsv su3_fieldsv.c)
add_su3_executable(su3_fieldsva su3_fieldsva.c)
add_su3_executable(time time.c)
add_su3_executable(time_gpu time_gpu.c)
add_su3_executable(soa_gpu soa_gpu.c)
add_su3_executable(soa_cpu soa_cpu.c)
 