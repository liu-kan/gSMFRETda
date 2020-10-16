cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
#find_package(CUDAToolkit REQUIRED)

project(gSMFRETda CUDA CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(Debug OFF CACHE BOOL "Shot debug flag")
set(DEFAULT_BUILD_TYPE "Release")
if(debug)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Debug is setted!")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Boost 1.65.0 COMPONENTS filesystem)
find_package(CUDA REQUIRED)
message(STATUS "Setting CUDA_curand_LIBRARY to '${CUDA_curand_LIBRARY}' .")

#set(CMAKE_CUDA_ARCH 3 5.2 6.1 7 7.5)
#set(CMAKE_CUDA_CODE 3 5.2 6.1 7 7.5)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G --expt-relaxed-constexpr --ptxas-options=-v  -Xptxas -dlcm=cg \  
      -gencode arch=compute_61,code=[sm_61,compute_61] \
      -gencode arch=compute_70,code=[sm_70,compute_70] \
      -gencode arch=compute_75,code=[sm_75,compute_75]")
      # -gencode arch=compute_75,code=sm_75 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_52,code=sm_52   -gencode arch=compute_70,code=sm_70 --maxrregcount=128
else ()
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-relaxed-constexpr --ptxas-options=-v -Xptxas -dlcm=cg \
       -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored \
       -gencode arch=compute_61,code=[sm_61,compute_61] \
      -gencode arch=compute_75,code=[sm_75,compute_75]")
       # -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_70,code=sm_70 --maxrregcount=128 
endif()
      #-gencode arch=compute_52,code=[sm_52,compute_52]  \
      #
      #-gencode arch=compute_70,code=[sm_70,compute_70] \



#set(link_stdc__fs "")
#if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
#  message( FATAL_ERROR "g++ need >=8, CMake will exit." )
#elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
#  message( STATUS "g++ 8.x, add -lstdc++fs" )
#  set(link_stdc__fs "-lstdc++fs")
#endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(boost_VERSION_MAJOR "1" CACHE STRING "boost's MAJOR version number")
set(boost_VERSION_MINOR "70" CACHE STRING "boost's MINOR version number")
set(boost_VERSION_PATCH "0" CACHE STRING "boost's PATCH version number")
set(rmm_VERSION_MAJOR "0" CACHE STRING "rmm's MAJOR version number")
set(rmm_VERSION_MINOR "15" CACHE STRING "rmm's MINOR version number")

set(boost_fname "${boost_VERSION_MAJOR}_${boost_VERSION_MINOR}_${boost_VERSION_PATCH}")
set(boost_ndir "${boost_VERSION_MAJOR}.${boost_VERSION_MINOR}.${boost_VERSION_PATCH}")
set(boostURL ${CMAKE_SOURCE_DIR}/3rdparty/boost_${boost_fname}.tar.bz2)

include(ExternalProject)
add_custom_target(third)

if (NOT EXISTS ${boostURL})
  #set(boostURL https://dl.bintray.com/boostorg/release/${boost_ndir}/source/boost_${boost_fname}.tar.bz2)
  set(boostURL http://mirrors.163.com/gentoo/distfiles/boost_${boost_fname}.tar.bz2)
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
  #GIT_REPOSITORY  https://git.liukan.org/mirrors/eigen.git
  GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/eigen
  GIT_SHALLOW 1
  GIT_TAG        3.3
  GIT_PROGRESS 1  
)
find_package(HighFive)
if(NOT HighFive_FOUND)
  message(STATUS "Don't worry!!! CMake will try to install dependencies for you! Maybe you need to input sudo password when you run make!")
    ExternalProject_Add(HighFive  
      GIT_REPOSITORY https://git.liukan.org/mirrors/HighFive.git
      GIT_SHALLOW 1
      GIT_PROGRESS 1
      CMAKE_ARGS -DHIGHFIVE_EXAMPLES=OFF -DHIGHFIVE_UNIT_TESTS=OFF
      PREFIX ${CMAKE_BINARY_DIR}/third/HighFive
      INSTALL_COMMAND sudo make install
      USES_TERMINAL_INSTALL true
      UPDATE_DISCONNECTED true
    )
    add_dependencies(third HighFive)
endif()
#    ExternalProject_Add(rmm  
#      GIT_REPOSITORY https://git.liukan.org/mirrors/rmm.git
#      GIT_PROGRESS 1
#      GIT_TAG origin/branch-${rmm_VERSION_MAJOR}.${rmm_VERSION_MINOR}
#      #PATCH_COMMAND patch -p0 < ${CMAKE_SOURCE_DIR}/3rdparty/rmm-0.9.patch 
#      PREFIX ${CMAKE_BINARY_DIR}/third/rmm
#      CMAKE_ARGS -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
#      INSTALL_COMMAND sudo make install
#      USES_TERMINAL_INSTALL true
#      UPDATE_DISCONNECTED true
#    )
#    add_dependencies(third rmm)
## https://anaconda.org/rapidsai/librmm/files
    add_custom_command(TARGET third
                      COMMAND pwd
                      PRE_BUILD
    )

#src/3rdparty/protobuf/args.pb.cc src/3rdparty/protobuf/args.pb.h

#find_package(Protobuf REQUIRED)
#include_directories(${Protobuf_INCLUDE_DIRS})
add_subdirectory(3rdparty)

execute_process(
    #COMMAND protoc --cpp_out=src 3rdparty/protobuf/args.proto
    COMMAND protoc --python_out=serv_py 3rdparty/protobuf/args.proto
    COMMAND gengetopt -i 3rdparty/gengetopt/cmdline.ggo --file-name=src/3rdparty/gengetopt/cmdline
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#file(COPY ${CMAKE_SOURCE_DIR}/src/3rdparty/protobuf DESTINATION ${CMAKE_SOURCE_DIR}/src)
file(COPY ${CMAKE_SOURCE_DIR}/serv_py/3rdparty/protobuf DESTINATION ${CMAKE_SOURCE_DIR}/serv_py)
file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/serv_py/3rdparty)
file(COPY ${CMAKE_SOURCE_DIR}/serv_py DESTINATION ${CMAKE_BINARY_DIR})

find_package(HDF5 REQUIRED)
find_package(NANOMSG QUIET)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(/usr/local/include)
include_directories(${CMAKE_BINARY_DIR}/third/eigen)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${nanomsg_INCDIR})
include_directories(${CMAKE_BINARY_DIR}/third/boost)
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${nanomsg_LIBDIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/compat)

add_executable(${PROJECT_NAME}
    #src/3rdparty/protobuf/args.pb.cc
    #src/3rdparty/protobuf/args.pb.h
    src/3rdparty/gengetopt/cmdline.c
    src/3rdparty/gengetopt/cmdline.h
    src/binom.cuh
    src/bitUbyte.hpp
    src/cuda_tools.hpp
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
    src/streamWorker.cpp
    src/streamWorker.hpp
    src/tools.cpp
    src/tools.hpp
    src/mrImp.cpp
    src/mrImp.hpp    
    )
    #src/rmm.hpp
target_link_libraries(${PROJECT_NAME} proto)

target_link_libraries(${PROJECT_NAME} -lpthread -lnanomsg -lboost_filesystem 
     ${HDF5_C_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY} ${Boost_LIBRARIES}) 
#${link_stdc__fs}
if (HighFive_FOUND)
  target_link_libraries(${PROJECT_NAME} HighFive)
endif()
  add_dependencies(${PROJECT_NAME} third)

