cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(gSMFRETda_tests LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 14)
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
    find_package(Boost REQUIRED)
else()
    FIND_PACKAGE(dlib REQUIRED)
    find_package(GSL REQUIRED)    
    find_package(Boost)
endif()
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT Boost_FOUND)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
endif()
   
###################################################################################################
# - compiler function
function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${CMAKE_TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${gSMFRETda_SOURCE_DIR}/src>")
    ### include_directories
    include_directories(${CUDA_INCLUDE_DIRS})
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS})
    elseif(NOT Boost_FOUND)
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_INCLUDE_DIRS_BOOST})
    else()
        target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
    endif()
    ### link_directories
    target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/compat)
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS} )
    elseif(NOT Boost_FOUND)
        target_link_directories(${CMAKE_TEST_NAME} PRIVATE ${CONAN_LIB_DIRS_BOOST})        
    endif()

    ### target_link_libraries
    target_link_libraries(${CMAKE_TEST_NAME}
                         ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY})    
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(${CMAKE_TEST_NAME} 
            ${CONAN_LIBS} ${CONAN_SYSTEM_LIBS}  )
    elseif(NOT Boost_FOUND)
        target_link_libraries(${CMAKE_TEST_NAME} 
            gtest gtest_main dl pthread ${CONAN_LIBS_BOOST} )
    else()
        TARGET_LINK_LIBRARIES(${CMAKE_TEST_NAME} 
            gtest gtest_main dl pthread ${Boost_LIBRARIES} )
    endif()

    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${gSMFRETda_BINARY_DIR}/gtests")
    set_target_properties(${CMAKE_TEST_NAME}  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${CMAKE_TEST_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        add_custom_target(virtual_target_${CMAKE_TEST_NAME})
        add_dependencies(${CMAKE_TEST_NAME} virtual_target_${CMAKE_TEST_NAME})
        add_custom_command(TARGET virtual_target_${CMAKE_TEST_NAME}
            COMMAND sed -i "s/-Wreturn-type//Ig"  tests/CMakeFiles/${CMAKE_TEST_NAME}.dir/flags.make
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Wreturn-type patch"
            PRE_BUILD
        )
    endif()
endfunction(ConfigureTest)


###################################################################################################
# - genrand tests
set(gen_rand_tests_SRC
	"${gSMFRETda_SOURCE_DIR}/tests/gen_rand_tests.cpp" 
    )
ConfigureTest(gen_rand_tests "${gen_rand_tests_SRC}")

add_library(genRand STATIC GenRandTest.hpp GenRandTest.cu ${gSMFRETda_SOURCE_DIR}/src/eigenhelper.cpp ${gSMFRETda_SOURCE_DIR}/src/eigenhelper.hpp
        hist.hpp hist.cpp) # boost histogram need to be compiled by gcc, no nvcc
target_link_libraries(genRand ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_curand_LIBRARY} ParamsTest)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(gen_rand_tests genRand ${CONAN_LIBS_GSL})
else()
    target_link_libraries(gen_rand_tests genRand GSL::gsl GSL::gslcblas dlib::dlib)
endif()

set(paraMatCheck_SRC
	"${gSMFRETda_SOURCE_DIR}/tests/paraMatCheck.cpp" 
    )
ConfigureTest(paraMatCheck_tests "${paraMatCheck_SRC}")
add_library(ParamsTest STATIC ParamsTest.hpp ParamsTest.cpp ${gSMFRETda_SOURCE_DIR}/src/eigenhelper.cpp ${gSMFRETda_SOURCE_DIR}/src/eigenhelper.hpp)
target_link_libraries(paraMatCheck_tests ParamsTest)

###################################################################################################
enable_testing()
