add_executable(project_video_export main.cpp)
target_link_libraries(project_video_export PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)
