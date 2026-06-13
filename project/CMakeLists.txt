add_executable(project main.cpp)


# target_include_directories(project PRIVATE "includes")


target_link_libraries(project PRIVATE fluid_simulation SDL3::SDL3 MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
