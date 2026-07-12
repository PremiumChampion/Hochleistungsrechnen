add_library(fluid_simulation src/field.cpp src/simulation.cpp)
# src/simulation_1d.cpp src/simulation_2d.cpp)
target_include_directories(fluid_simulation PUBLIC "includes")
target_link_libraries(fluid_simulation PRIVATE MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
