# Lattice Boltzmann Method (LBM) Fluid Simulation

This repository contains a 2D Lattice Boltzmann Method (LBM) fluid simulation parallelized with Kokkos (for GPU/CPU acceleration) and MPI (for distributed memory scaling). It includes interactive visualization via SDL3 and offline video generation using FFmpeg.

## Prerequisites

Before compiling, ensure your system has the following dependencies installed:

*   **C++ Compiler**: A compiler supporting C++20 (e.g., GCC 10+, Clang 11+, or MSVC 2022+).
*   **CMake**: Version 3.14 or higher.
*   **MPI**: An MPI implementation such as OpenMPI or MPICH.
*   **FFmpeg**: Required at runtime for video-export targets to pipe raw pixel data into MP4 containers.
*   **SDL3 System Libraries**: Depending on your Linux distribution, you may need development packages for your window manager (e.g., Wayland or X11 development headers) to run the interactive targets. [deps](https://github.com/libsdl-org/SDL/blob/main/docs/README-linux.md#build-dependencies)

### Installing Prerequisites (Ubuntu/Debian example)
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential openmpi-bin libopenmpi-dev ffmpeg libx11-dev libwayland-dev
```

## Building the Project

The project uses CMake to fetch and configure external dependencies (Eigen3, Kokkos, and SDL3) automatically.

```bash
# Configure the build directory
# USE_CUDA=ON 
#       if Kokkos is installed on your system this does not have any effect
#       enable cuda with the AMPERE86 architecture
#       change these values in the CMakeLists.txt for your build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# Compile all targets
cmake --build build -j$(nproc)
```

## Running the Simulation

The executables are placed in subdirectories under `build/Release/` (or `build/Debug/` depending on your build type). Below are the commands to run the key components.

### 1. Interactive Canvas (SDL3)
Draw walls, sources, and sinks in real-time. Change view modes (speed, direction, density) with your keyboard:
```bash
# Run sequentially
./build/Release/interactive/interactive

# Or run in parallel with MPI (example: 4 ranks)
mpirun -np 4 ./build/Release/interactive/interactive
```

### 2. Video Export Presentation (FFmpeg)
Simulates a fluid flow passing through a complex obstacle course and generates three high-definition video files (`output_1080p_direction.mp4`, `output_1080p_density.mp4`, and `output_1080p_speed.mp4`):
```bash
mpirun -np 4 ./build/Release/project_video_export_presentation/project_video_export_presentation
```

### 3. Lid-Driven Cavity Simulation & Visualization
Runs a standard lid-driven cavity validation setup:
```bash
# Headless convergence test (saves CSV files)
mpirun -np 4 ./build/Release/milestone05/milestone05

# Real-time visual feedback
mpirun -np 4 ./build/Release/milestone05/milestone05_visualisation
```

### 4. Running Unit Tests
Validate the implementation of the streaming, collision, and macroscopic calculations:
```bash
ctest --test-dir build --output-on-failure
```