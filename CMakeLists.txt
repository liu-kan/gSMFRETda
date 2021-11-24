cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gSMFRETda 
        VERSION 0.8.7.201204
        LANGUAGES CXX C CUDA)
#set(CMAKE_CXX_STANDARD 14) # enable std::byte
# gcc now has bug if -std=c++17 is used with Boost::histogram Oct. 2020
# https://github.com/boostorg/histogram/issues/290
# fixed in 1.75@boostorg/histogram
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CMakeFindDependencyMacro)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.17.0") 
    find_dependency(CUDAToolkit)
endif()
# Auto-detect available GPU compute architectures
include(${CMAKE_SOURCE_DIR}/cmake/SetGPUArchs.cmake)
message(STATUS "Building with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
#enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

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

option(pserver "whether auto build pySMFRETda" ON)
option(proto "whether update proto in pySMFRETda SOURCE_DIR" OFF)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
    if(DEFINED ENV{patch})
        set(win_patch $ENV{patch})
    else()
        set(win_patch "C:/Program Files/Git/usr/bin/patch.exe")
    endif()  
endif()
find_package(Boost 1.71.0 COMPONENTS filesystem)
find_package(CUDA REQUIRED)
message(STATUS "Setting CUDA_curand_LIBRARY to '${CUDA_curand_LIBRARY}' .")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(link_stdc__fs "")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
        message( FATAL_ERROR "g++ need >=8, CMake will exit." )
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        message( STATUS "g++ 8.x, add -lstdc++fs" )
        set(link_stdc__fs "stdc++fs")
    #elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    #    message( STATUS "g++ 11, use g++-10" )
    #    set(CMAKE_C_COMPILER gcc-10 CACHE PATH "" FORCE)
    #    set(CMAKE_CXX_COMPILER g++-10 CACHE PATH "" FORCE)  
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-unused-result -s")
endif()

include(ExternalProject)
add_custom_target(third)

include(FetchContent)
FetchContent_Populate(
  eigen
  GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/eigen
  GIT_SHALLOW    1
  GIT_TAG        3.3
  GIT_PROGRESS 1
  UPDATE_DISCONNECTED true
)

if(pserver)
FetchContent_Populate(
  pySMFRETda
  GIT_REPOSITORY  https://gitlab.com/liu-kan/pySMFRETda.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/pySMFRETda
  GIT_SHALLOW 1
  GIT_TAG        origin/main
  GIT_PROGRESS 1  
  UPDATE_DISCONNECTED true
)
endif()
FetchContent_Populate(
  HighFive
  GIT_REPOSITORY  https://gitlab.com/liu-kan/HighFive.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/third/HighFive
  GIT_SHALLOW 1
  GIT_PROGRESS 1
  UPDATE_DISCONNECTED true
)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
add_custom_command(TARGET third
    COMMAND set win_patch="${win_patch}" & set eigen_patch="${CMAKE_SOURCE_DIR}/3rdparty/patches/eigenHalfH.patch" & ${CMAKE_SOURCE_DIR}/3rdparty/patches/patches_apply.bat
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third/eigen
    COMMENT "Patching Eigen"
    PRE_BUILD
)
endif()

add_subdirectory(3rdparty)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
execute_process(
    COMMAND gengetopt -i 3rdparty/gengetopt/cmdline.ggo --file-name=src/3rdparty/gengetopt/cmdline
    COMMAND gengetopt -i 3rdparty/gengetopt/call_gSMFRETda.ggo --file-name=src/3rdparty/gengetopt/call_gSMFRETda
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
endif()
if(EXISTS ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga)
#[[
  if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(CMAKE_TOOLCHAIN_FILE)
        message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
        string(FIND ${CMAKE_TOOLCHAIN_FILE} "/scripts/buildsystems/vcpkg.cmake" VCPKG_INSTALLATION_ROOT_len)
        message(STATUS "VCPKG_INSTALLATION_ROOT_len: ${VCPKG_INSTALLATION_ROOT_len}")
        string(SUBSTRING ${CMAKE_TOOLCHAIN_FILE} 0 ${VCPKG_INSTALLATION_ROOT_len} VCPKG_INSTALLATION_ROOT)
        message(STATUS "VCPKG_INSTALLATION_ROOT: ${VCPKG_INSTALLATION_ROOT}")    
        execute_process(
            COMMAND ${VCPKG_INSTALLATION_ROOT}/installed/x64-windows/tools/protoc --python_out=${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga 3rdparty/protobuf/args.proto
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            ECHO_OUTPUT_VARIABLE
            ECHO_ERROR_VARIABLE
        )
    else()
        execute_process(            
            COMMAND ${CONAN_BIN_DIRS_PROTOBUF/protoc --python_out=${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga 3rdparty/protobuf/args.proto            
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            ECHO_OUTPUT_VARIABLE
            ECHO_ERROR_VARIABLE
        )    
    endif()
  else()
      execute_process(
          COMMAND protoc --python_out=${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga 3rdparty/protobuf/args.proto
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
  endif()
]]
  #file(COPY ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga/3rdparty/protobuf DESTINATION ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga)
  file(COPY ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga DESTINATION ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga)
    if(EXISTS ${CMAKE_SOURCE_DIR}/../pySMFRETda/serv_pdaga AND proto)
        file(COPY ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga/3rdparty/protobuf DESTINATION ${CMAKE_SOURCE_DIR}/../pySMFRETda/serv_pdaga)
    endif()  
  file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/pySMFRETda/serv_pdaga/3rdparty)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  include_directories(${CONAN_INCLUDE_DIRS})  
else()
  find_package(HDF5 REQUIRED)
  include_directories(${HDF5_INCLUDE_DIRS})
  #find_package(NANOMSG QUIET)
  find_package(nng CONFIG REQUIRED)
endif()

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/third/HighFive/include)
include_directories(${CMAKE_SOURCE_DIR})

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    include_directories(${CONAN_INCLUDE_DIRS_BOOST})    
elseif(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})    
    include_directories(/usr/local/include)
else()
    include_directories(/usr/local/include)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include_directories(${CONAN_INCLUDE_DIRS_BOOST})    
endif()

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/compat)

include_directories(${CMAKE_BINARY_DIR}/third/eigen)    
if (CMAKE_SYSTEM_NAME MATCHES "Windows")    
    link_directories(${CONAN_LIB_DIRS_BOOST})
elseif(Boost_FOUND)
else()
    link_directories(call_${PROJECT_NAME} ${CONAN_LIB_DIRS_BOOST})    
endif()

#if(MSVC)
#           set(CMAKE_EXE_LINKER_FLAGS
#          "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:msvcrt.lib")
#endif()

add_executable(call_${PROJECT_NAME}
    src/3rdparty/gengetopt/call_gSMFRETda.c
    src/3rdparty/gengetopt/call_gSMFRETda.h
    src/call_main.cpp
    resources/gSMFRETda.rc
    )
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(call_${PROJECT_NAME} 
        ${CONAN_LIBS_BOOST} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
elseif(Boost_FOUND)
    target_link_libraries(call_${PROJECT_NAME} 
        ${Boost_LIBRARIES} boost_filesystem dl ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} pthread) 
else()
    target_link_libraries(call_${PROJECT_NAME} 
        ${CONAN_LIBS_BOOST} dl ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} pthread) 
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    link_directories(${CONAN_LIB_DIRS})
elseif(NOT Boost_FOUND)
    link_directories(${CONAN_LIB_DIRS_BOOST})    
endif()

add_executable(${PROJECT_NAME}
    src/3rdparty/gengetopt/cmdline.c
    src/3rdparty/gengetopt/cmdline.h
    src/binom.cuh
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
    resources/gSMFRETda.rc
    )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(call_${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET call_${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
#set_property(TARGET proto PROPERTY CUDA_ARCHITECTURES 61 70 75 72-virtual)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS_HDF5} ${CONAN_LIBS_BOOST} ${CONAN_LIBS_NNG})
elseif(Boost_FOUND)
    target_link_libraries(${PROJECT_NAME} ${HDF5_C_LIBRARIES} boost_filesystem 
          pthread nng::nng ${link_stdc__fs} ${Boost_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME} ${HDF5_C_LIBRARIES} ${CONAN_LIBS_BOOST} ${link_stdc__fs} pthread nng::nng)
endif()
target_link_libraries(${PROJECT_NAME} proto)
target_link_libraries(${PROJECT_NAME} 
     ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY})

#if (HighFive_FOUND)
#  target_link_libraries(${PROJECT_NAME} HighFive)
#endif()
  add_dependencies(${PROJECT_NAME} third)

option(BUILD_TESTS "Configure CMake to build tests" OFF)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif(BUILD_TESTS)