cmake_minimum_required(VERSION 3.26)
project(OwnCUDASIFT CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif (NOT CUDA_FOUND)

set(CMAKE_CUDA_STANDARD 17)

set(cuda_sources
        cudaImage.cu
        cudaImage.h
        cudaSift.h
        cudaSiftD.h
        cudaSiftH.cu
        cudaSiftH.h
        cudautils.h
        matching.cu
)
set(sources geomFuncs.cpp
        mainSift.cpp
        utility.cpp
        utility.h
)

add_executable(OwnCUDASIFT ${cuda_sources} ${sources})

set_target_properties(OwnCUDASIFT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(OwnCUDASIFT ${OpenCV_LIBS})