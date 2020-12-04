cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(gSMFRETda_tests LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CMakeFindDependencyMacro)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.17.0") 
    find_dependency(CUDAToolkit)
endif()
# Auto-detect available GPU compute architectures
include(${CMAKE_SOURCE_DIR}/cmake/SetGPUArchs.cmake)
message(STATUS "Building with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
enable_language(CUDA)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    include_directories(${CONAN_INCLUDE_DIRS_DLIB})
    include_directories(${CONAN_INCLUDE_DIRS_GTEST})
else()
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include_directories(${CONAN_INCLUDE_DIRS_BOOST})
    include_directories(${nanomsg_INCDIR})
endif()

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/compat)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    link_directories(${CONAN_LIB_DIRS_DLIB})
    link_directories(${CONAN_LIB_DIRS_GTEST})
else()
    link_directories(${nanomsg_LIBDIR})
    
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(call_${PROJECT_NAME} 
        ${CONAN_LIBS_GTEST} ${CONAN_LIBS_DLIB} ${CONAN_SYSTEM_LIBS_DLIB} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
else()
    target_link_libraries(call_${PROJECT_NAME} 
        ${CONAN_LIBS_BOOST} -ldl ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} -lpthread) 
endif()

set_target_properties(call_${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET call_${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
#set_property(TARGET proto PROPERTY CUDA_ARCHITECTURES 61 70 75 72-virtual)
add_compile_options(${PROJECT_NAME} 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS_HDF5} ${CONAN_LIBS_BOOST} ${CONAN_LIBS_NANOMSG})
elseif(Boost_FOUND)
    target_link_libraries(${PROJECT_NAME} ${HDF5_C_LIBRARIES} -lboost_filesystem 
          -lpthread -lnanomsg ${link_stdc__fs} ${Boost_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME} ${HDF5_C_LIBRARIES} ${CONAN_LIBS_BOOST} ${link_stdc__fs} -lpthread -lnanomsg)
endif()
target_link_libraries(${PROJECT_NAME} proto)
target_link_libraries(${PROJECT_NAME} 
     ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY})

###################################################################################################
# - compiler function
function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
  add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
  set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${gSMFRETda_SOURCE_DIR}>")
  target_link_libraries(${CMAKE_TEST_NAME} GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
                        pthread )
  
  set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY "${gSMFRETda_BINARY_DIR}/gtests")
  target_compile_definitions(${CMAKE_TEST_NAME} PUBLIC
                        "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${gSMFRETda_LOGGING_LEVEL}")    
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - uvector tests
set(DEVICE_UVECTOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_uvector_tests.cpp")
ConfigureTest(DEVICE_UVECTOR_TEST "${DEVICE_UVECTOR_TEST_SRC}")


###################################################################################################
enable_testing()