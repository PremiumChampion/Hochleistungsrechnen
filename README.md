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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON

# Compile all targets
cmake --build build -j$(nproc)
```

Here are the commands for different environment:     
       
 ### 1. NOGPU      

```bash        
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNO_GLOBAL_KOKKOS=ON
```

### 2. Local RTX 3050 Ti
 
```bash        
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON 
```   
       
### 3. Cluster A100 + H100
       
```bash
# A100 only
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON 

# H100 only  
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON 
```       

## Running the Simulation

The executables are placed in subdirectories under `build/Release/` (or `build/Debug/` depending on your build type). Below are the commands to run the key components.

### 1. Interactive Canvas (SDL3)
Draw walls, sources, and sinks in real-time. Change view modes (speed, direction, density) with your keyboard:
```bash
# Run sequentially
./build/Release/interactive

# Or run in parallel with MPI (example: 4 ranks)
mpirun -np 4 ./build/Release/interactive
```

### 2. Video Export Presentation (FFmpeg)
Simulates a fluid flow passing through a complex obstacle course and generates three high-definition video files (`output_1080p_direction.mp4`, `output_1080p_density.mp4`, and `output_1080p_speed.mp4`):
```bash
mpirun -np 4 ./build/Release/project_video_export_presentation
```

### 3. Lid-Driven Cavity Simulation & Visualization
Runs a standard lid-driven cavity validation setup:
```bash
# Headless convergence test (saves CSV file)
mpirun -np 4 ./build/Release/milestone05

# Real-time visual feedback
mpirun -np 4 ./build/Release/milestone05_visualisation
```

### 4. Running Unit Tests
Validate the implementation of the streaming, collision, and macroscopic calculations:
```bash
ctest --test-dir build --output-on-failure
```

### Valid executables

| executable                        | funciton                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| benchmark                         | Measures the performance (in MLUPS) and tracks compute time versus MPI communication time                    |
| interactive                       | A real-time interactive simulation, adjust simulation parameters, and toggle between visualisations.         |
| lbm_tests                         | The unit testing suite (using GoogleTest) that validates core LBM components                                 |
| milestone04                       | Validates fluid physics by simulating a shear wave decay.                                                    |
| milestone05                       | A headless standard lid-driven cavity simulation that runs until the fluid flow converges to a steady state. |
| milestone05_video_export          | Runs a high-resolution lid-driven cavity simulation headlessly and exports a video from the results.         |
| milestone05_visualisation         | A real-time visualization of the lid-driven cavity flow, allowing for live observation of the simulation.    |
| milestone06                       | Demonstrates and benchmarks distributed memory parallelization.                                              |
| project                           | A basic, real-time SDL3 visualization of the general LBM fluid simulation.                                   |
| project_video_export              | Runs a standard simulation and saves the velocity magnitude (speed) to a video file.                         |
| project_video_export_presentation | Simulates a complex obstacle course and generates three distinct videos: direction, density, and speed.      |
| serial                            | Demonstrates simulation serialization.                                                                       |
| validation                        | Runs on a single MPI rank to perform strict physics validation.                                              |


### HPC Benchmarking & Slurm Integration
The project includes a comprehensive suite for evaluating MPI/Kokkos scaling on HPC clusters (like bwUniCluster). The pipeline handles job submission, data extraction, and plotting.

1. **Submit Jobs:** Navigate to the benchmark folder and submit the strong and weak scaling jobs.
   ```bash
   cd performance_benchmarks
   ./submit-benchmarks.sh
   ```
2. **Parse Results:** Once the Slurm jobs finish, extract the metrics from the `.out` logs into a clean CSV file.
   ```bash
   python parse_slurm.py
   ```
3. **Visualize:** Generate performance and runtime graphs based on the CSV data.
   ```bash
   python visualize_slurm.py
   ```

## Python - Dependencies & Visualization

### Prerequisites
You will need Python 3 and a few standard data science libraries:
```bash
pip install pandas numpy matplotlib
```

### Generating Validation Plots

Several executables export physics data to CSV files. You can generate plots to validate the physics by running the corresponding Python scripts after the simulation finishes:

```bash
# Validate shear wave decay and kinematic viscosity
mpirun -np 1 ./build/Release/milestone04
python milestone04/plot.py

# Plot lid-driven cavity streamlines and velocity profiles
mpirun -np 4 ./build/Release/milestone05
python milestone05/plot.py

# Plot mass conservation and density relaxation 
./build/Release/validation
python validation/plot.py
```

---

# Notes

All packages to install for Ubuntu 24 LTS:

```bash 
sudo apt-get update -y 
sudo apt-get install -y cmake ninja-build openmpi-bin libopenmpi-dev cmake build-essential \
  openmpi-bin libopenmpi-dev ffmpeg libx11-dev libwayland-dev build-essential git make 
  pkg-config cmake ninja-build gnome-desktop-testing libasound2-dev libpulse-dev \
  libaudio-dev libfribidi-dev libjack-dev libsndio-dev libx11-dev libxext-dev \
  libxrandr-dev libxcursor-dev libxfixes-dev libxi-dev libxss-dev libxtst-dev \
  libxkbcommon-dev libdrm-dev libgbm-dev libgl1-mesa-dev libgles2-mesa-dev \
  libegl1-mesa-dev libdbus-1-dev libibus-1.0-dev libudev-dev libthai-dev libusb-1.0-0-dev \
  libpipewire-0.3-dev libwayland-dev libdecor-0-dev liburing-dev
```