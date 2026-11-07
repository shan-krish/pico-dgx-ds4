# The vision tower (vision.h): DeepSeek's ViT and aligner, run by every rank on a request's images.
# cuBLAS (static) does its GEMMs.
find_package(CUDAToolkit REQUIRED)

add_library(dsvision STATIC vision.cu)
target_include_directories(dsvision PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dsvision PUBLIC dsw_host CUDA::cublas_static CUDA::cudart_static)

# one image through the tower against the reference's output (tools/vit_ref.py)
add_executable(vit_check tests/vit_check.cpp)
target_link_libraries(vit_check PRIVATE dsvision)
