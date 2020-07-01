cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gSMFRETda CUDA CXX C)
set(CMAKE_CXX_STANDARD 14)

set(boost_VERSION_MAJOR "1" CACHE STRING "boost's MAJOR version number")
set(boost_VERSION_MINOR "70" CACHE STRING "boost's MINOR version number")
set(boost_VERSION_PATCH "0" CACHE STRING "boost's PATCH version number")
set(rmm_VERSION_MAJOR "0" CACHE STRING "rmm's MAJOR version number")
set(rmm_VERSION_MINOR "9" CACHE STRING "rmm's MINOR version number")

set(boost_fname "${boost_VERSION_MAJOR}_${boost_VERSION_MINOR}_${boost_VERSION_PATCH}")
set(boost_ndir "${boost_VERSION_MAJOR}.${boost_VERSION_MINOR}.${boost_VERSION_PATCH}")
set(boostURL ${CMAKE_SOURCE_DIR}/3rdparty/boost_${boost_fname}.tar.bz2)

include(ExternalProject)
add_custom_target(third)

if (NOT EXISTS ${boostURL})
  set(boostURL https://dl.bintray.com/boostorg/release/${boost_ndir}/source/boost_${boost_fname}.tar.bz2)
  message(STATUS "CMake will download boost from ${boostURL} for you!!!!!")
endif()

include(FetchContent)
FetchContent_Populate(
  boost
  URL ${boostURL}
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/boost
)
FetchContent_Populate(
  eigen
  GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/eigen
  GIT_SHALLOW 1
  GIT_PROGRESS 1  
)
FetchContent_Populate(
  cxxopts
  GIT_REPOSITORY  https://github.com/jarro2783/cxxopts.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/cxxopts
  GIT_SHALLOW 1
  GIT_PROGRESS 1  
)

ExternalProject_Add(HighFive  
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_SHALLOW 1
  GIT_PROGRESS 1
  CMAKE_ARGS -DHIGHFIVE_EXAMPLES=OFF -DHIGHFIVE_UNIT_TESTS=OFF
  PREFIX ${CMAKE_BINARY_DIR}/third/HighFive
  INSTALL_COMMAND sudo make install
  USES_TERMINAL_INSTALL true
)

ExternalProject_Add(rmm  
  GIT_REPOSITORY https://github.com/rapidsai/rmm.git
  GIT_PROGRESS 1
  GIT_TAG origin/branch-${rmm_VERSION_MAJOR}.${rmm_VERSION_MINOR}
  PATCH_COMMAND patch -p0 < ${CMAKE_SOURCE_DIR}/3rdparty/rmm-0.9.patch 
  PREFIX ${CMAKE_BINARY_DIR}/third/rmm
  CMAKE_ARGS -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF
  INSTALL_COMMAND sudo make install
  USES_TERMINAL_INSTALL true
)

add_dependencies(third HighFive)
add_dependencies(third rmm)
execute_process (
    COMMAND protoc --cpp_out=src protobuf/args.proto
    COMMAND protoc --python_out=serv_py protobuf/args.proto
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)
find_package(NANOMSG QUIET)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/protobuf)
include_directories(/usr/local/include)
include_directories(${CMAKE_BINARY_DIR}/third/eigen)
include_directories(${CMAKE_BINARY_DIR}/third/cxxopts/include)
include_directories(${nanomsg_INCDIR})
include_directories(${CMAKE_BINARY_DIR}/third/boost)

link_directories(${nanomsg_LIBDIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/compat)

add_executable(${PROJECT_NAME}
    src/protobuf/args.pb.cc
    src/protobuf/args.pb.h
    src/binom.cuh
    src/bitUbyte.hpp
    src/cuda_tools.hpp
    src/cuList.cuh
    src/eigenhelper.cpp
    src/eigenhelper.hpp
    src/gen_rand.cuh
    src/gpuWorker.cpp
    src/gpuWorker.hpp
    src/loadHdf5.cpp
    src/loadHdf5.hpp
    src/main.cpp
    src/mc.cu
    src/mc.hpp
    src/rmm.hpp
    src/streamWorker.cpp
    src/streamWorker.hpp
    src/tools.cpp
    src/tools.hpp
    )

add_dependencies(${PROJECT_NAME} third)

target_link_libraries(${PROJECT_NAME} -lrmm -lpthread -lnanomsg -lprotobuf -lnanomsg
     ${HDF5_C_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY} )