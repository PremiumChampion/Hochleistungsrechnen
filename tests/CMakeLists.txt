add_executable(lbm_tests test_lbm.cpp)

# Link against the core library, GoogleTest, Kokkos, and MPI
target_link_libraries(lbm_tests PRIVATE fluid_simulation GTest::gtest MPI::MPI_CXX Kokkos::kokkos)

# Register the tests with CTest
include(GoogleTest)
gtest_discover_tests(lbm_tests)