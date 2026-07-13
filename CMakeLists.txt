cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
project(my-fantastic-simulation-code LANGUAGES C CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)

    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

# Dependencies: MPI, Eigen3, Kokkos
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    add_compile_definitions(WITH_MPI)
    message(STATUS "MPI found")
endif()

find_package(Eigen3 5.0.1)
if(NOT Eigen3_FOUND)
    include(FetchContent)

    FetchContent_Declare(
      eigen3
      URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    )
    set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

find_package(Kokkos 4.7.03)
if(NOT Kokkos_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      kokkos
      URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.03.tar.gz
    )
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)

    # Enable modern CMake CUDA compilation
    # set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "Compile as CUDA" FORCE)
    # set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
    # set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Enable DEBUG" FORCE)
    # set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "RTX 3070 Architecture" FORCE)
    
    option(USE_CUDA "Enable CUDA backend for Kokkos" OFF)
    if(USE_CUDA)
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda" FORCE)
        # Target RTX 3070 GPU (Ampere86)
        set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "RTX 3070 Architecture" FORCE)
    else()
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Serial Backend" FORCE)
    endif()

    FetchContent_MakeAvailable(kokkos)

    if(USE_CUDA)
      kokkos_compilation(GLOBAL)
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")


include(FetchContent)

set(SDL_SHARED OFF)
set(SDL_STATIC ON)
set(BUILD_SHARED_LIBS OFF)
set(SDLIMAGE_DEPS_SHARED OFF)
set(SDLNET_DEPS_SHARED OFF)

FetchContent_Declare(
  sdl3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG        badbf8da4ee72b3ef599c721ffc9899e8d7c8d90 # release-3.2.26
)

FetchContent_MakeAvailable(sdl3)

enable_testing()
add_subdirectory(fluid_simulation)
add_subdirectory(project)
add_subdirectory(serial)
add_subdirectory(project_video_export)
add_subdirectory(project_video_export_presentation)
add_subdirectory(milestone04)
add_subdirectory(milestone05)
add_subdirectory(milestone06)
add_subdirectory(interactive)
add_subdirectory(benchmark)
add_subdirectory(tests)