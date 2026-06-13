add_executable(serial main.cpp)
target_link_libraries(serial PRIVATE fluid_simulation SDL3::SDL3 MPI::MPI_CXX Kokkos::kokkos)