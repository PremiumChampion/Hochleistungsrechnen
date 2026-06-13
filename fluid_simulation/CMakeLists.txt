add_library(fluid_simulation src/field.cpp src/simulation.cpp)
target_include_directories(fluid_simulation PUBLIC "includes")
target_link_libraries(fluid_simulation PRIVATE MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
