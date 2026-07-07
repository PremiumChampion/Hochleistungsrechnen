add_executable(benchmark main.cpp)
target_link_libraries(benchmark PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)