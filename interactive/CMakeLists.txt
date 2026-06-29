add_executable(interactive main.cpp)
target_link_libraries(interactive PRIVATE fluid_simulation SDL3::SDL3 MPI::MPI_CXX Kokkos::kokkos)