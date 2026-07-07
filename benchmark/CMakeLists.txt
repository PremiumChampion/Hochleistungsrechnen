add_executable(benachmark main.cpp)
target_link_libraries(benachmark PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)