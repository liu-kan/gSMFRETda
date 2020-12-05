cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(gSMFRETda_tests LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CMakeFindDependencyMacro)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.17.0") 
    find_dependency(CUDAToolkit)
endif()
# Auto-detect available GPU compute architectures
include(${gSMFRETda_SOURCE_DIR}/cmake/SetGPUArchs.cmake)
message(STATUS "Building TESTs with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
#enable_language()

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
else()
    FIND_PACKAGE(dlib REQUIRED)
    find_package(Boost 1.70.0 EXACT)
endif()
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT Boost_FOUND)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
endif()
if (${DLIB_USE_CUDA})
  add_definitions("-DDLIB_USE_CUDA")
endif()

###################################################################################################
# - compiler function
function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    #set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    #target_include_directories(${CMAKE_TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${gSMFRETda_SOURCE_DIR}/src>")
    ### include_directories
    include_directories(${CUDA_INCLUDE_DIRS})
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        #target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS_DLIB})
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS_GTEST})
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
    elseif(NOT Boost_FOUND)
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
    else()
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
    endif()
    ### link_directories
    target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/compat)
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        #target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS_DLIB})
        target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS_GTEST})
        target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS_BOOST})
    elseif(NOT Boost_FOUND)
        target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS_BOOST})        
    endif()

    ### target_link_libraries
    target_link_libraries(${CMAKE_TEST_NAME}
                         ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})    
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(${CMAKE_TEST_NAME} 
            ${CONAN_LIBS_GTEST} ${Boost_LIBRARIES})
            #${CONAN_LIBS_DLIB} ${CONAN_SYSTEM_LIBS_DLIB} 
    elseif(NOT Boost_FOUND)
        target_link_libraries(${CMAKE_TEST_NAME} 
            gtest gtest_main dl pthread ${CONAN_LIBS_BOOST}) 
            #dlib::dlib 
    else()
        TARGET_LINK_LIBRARIES(${CMAKE_TEST_NAME} 
            gtest gtest_main dl pthread ${Boost_LIBRARIES}) 
            #dlib::dlib
    endif()

    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${gSMFRETda_BINARY_DIR}/gtests")
    set_target_properties(${CMAKE_TEST_NAME}  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${CMAKE_TEST_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - genrand tests
set(gen_rand_tests_SRC
	"${gSMFRETda_SOURCE_DIR}/tests/gen_rand_tests.cu" 
    )
ConfigureTest(gen_rand_tests "${gen_rand_tests_SRC}")

###################################################################################################
enable_testing()
