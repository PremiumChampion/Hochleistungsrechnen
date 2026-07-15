add_executable(validation main.cpp)
target_link_libraries(validation PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)
