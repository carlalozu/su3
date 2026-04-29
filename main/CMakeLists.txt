
macro(add_su3_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE su3openmp)
    target_link_libraries(${name} PRIVATE su3lib)
endmacro()

# add_su3_executable(su3_alg su3_alg.c)
# add_su3_executable(su3_fields su3_fields.c)
# add_su3_executable(su3_fieldsv su3_fieldsv.c)
# add_su3_executable(su3_fieldsva su3_fieldsva.c)
# add_su3_executable(time time.c)
if(ENABLE_GPU_OFFLOAD)
    add_su3_executable(soa_gpu_openmp soa_gpu_openmp.c)
    add_su3_executable(aos_gpu_openmp aos_gpu_openmp.c)
    add_su3_executable(time_gpu time_gpu.c)
    add_su3_executable(aos_cpu aos_cpu.c)
endif()

if(ENABLE_CUDA)
    add_executable(soa_gpu_cuda soa_gpu_cuda.cu)
    target_link_libraries(soa_gpu_cuda PRIVATE su3cudakernel)
    set_target_properties(soa_gpu_cuda PROPERTIES CUDA_ARCHITECTURES "90")

    add_executable(aos_gpu_cuda aos_gpu_cuda.cu)
    target_link_libraries(aos_gpu_cuda PRIVATE su3cudakernel)
    set_target_properties(aos_gpu_cuda PROPERTIES CUDA_ARCHITECTURES "90")
endif()

if(ENABLE_KOKKOS)
    add_executable(soa_gpu_kokkos soa_gpu_kokkos.cpp)
    target_link_libraries(soa_gpu_kokkos PRIVATE su3kokkoskernel)

    add_executable(aos_gpu_kokkos aos_gpu_kokkos.cpp)
    target_link_libraries(aos_gpu_kokkos PRIVATE su3kokkoskernel)

    if(Kokkos_ENABLE_CUDA)
        set_source_files_properties(soa_gpu_kokkos.cpp PROPERTIES LANGUAGE CUDA)
        set_target_properties(soa_gpu_kokkos PROPERTIES CUDA_ARCHITECTURES "90")

        set_source_files_properties(aos_gpu_kokkos.cpp PROPERTIES LANGUAGE CUDA)
        set_target_properties(aos_gpu_kokkos PROPERTIES CUDA_ARCHITECTURES "90")

        target_compile_options(su3kokkoskernel PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        )
    endif()
endif()
