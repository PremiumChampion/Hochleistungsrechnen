add_executable(milestone05 main.cpp)
target_link_libraries(milestone05 PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)

add_executable(milestone05_visualisation visualisation.cpp)
target_link_libraries(milestone05_visualisation PRIVATE fluid_simulation SDL3::SDL3 MPI::MPI_CXX Kokkos::kokkos)

add_executable(milestone05_video_export video_export.cpp)
target_link_libraries(milestone05_video_export PRIVATE fluid_simulation MPI::MPI_CXX Kokkos::kokkos)