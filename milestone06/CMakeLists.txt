add_executable(milestone06 main.cpp)
target_link_libraries(milestone06 PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)