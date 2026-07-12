#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "field.h"

// -----------------------------------------------------------------------------
// Test 1: compute_velocity
// Checks whether macroscopic fields (density and velocity) are accurately
// calculated from the probability distribution functions (f).
// -----------------------------------------------------------------------------
TEST(LBMTest, ComputeVelocity) {
    int Nx = 3, Ny = 3;
    ScalarField2D rho("rho", Nx + 2, Ny + 2);
    VectorField2D v("v", Nx + 2, Ny + 2);
    DistFuncD2Q9 f("f", Nx + 2, Ny + 2);

    auto h_f = Kokkos::create_mirror_view(f);
    
    for (int y = 1; y <= Ny; ++y) {
        for (int x = 1; x <= Nx; ++x) {
            h_f(x, y, 0) = 1.0; 
            h_f(x, y, 1) = 1.0; // cx = 1, cy = 0
            for (int i = 2; i < 9; ++i) h_f(x, y, i) = 0.0;
        }
    }
    Kokkos::deep_copy(f, h_f);

    compute_velocity(rho, v, f, Nx, Ny);
    Kokkos::fence();

    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_v = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_v, v);

    // Sum(f) = 1.0 + 1.0 = 2.0
    EXPECT_DOUBLE_EQ(h_rho(2, 2), 2.0);
    // vx = Sum(f * cx) / rho = (1.0 * 1) / 2.0 = 0.5
    EXPECT_DOUBLE_EQ(h_v(2, 2, 0), 0.5);
    EXPECT_DOUBLE_EQ(h_v(2, 2, 1), 0.0);
}

// -----------------------------------------------------------------------------
// Test 2: collide
// Validates whether the BGK collision operator appropriately relaxes a non-
// equilibrium fluid state towards equilibrium.
// -----------------------------------------------------------------------------
TEST(LBMTest, Collide) {
    int Nx = 3, Ny = 3;
    ScalarField2D rho("rho", Nx + 2, Ny + 2);
    VectorField2D v("v", Nx + 2, Ny + 2);
    DistFuncD2Q9 f("f", Nx + 2, Ny + 2);

    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_v = Kokkos::create_mirror_view(v);
    auto h_f = Kokkos::create_mirror_view(f);
    
    for (int y = 1; y <= Ny; ++y) {
        for (int x = 1; x <= Nx; ++x) {
            h_rho(x, y) = 1.0;
            h_v(x, y, 0) = 0.0;
            h_v(x, y, 1) = 0.0;
            h_f(x, y, 0) = 0.5; // Incorrect distribution state
            for (int i = 1; i < 9; ++i) h_f(x, y, i) = 0.0;
        }
    }
    Kokkos::deep_copy(rho, h_rho);
    Kokkos::deep_copy(v, h_v);
    Kokkos::deep_copy(f, h_f);

    // With omega = 1.0, f -> f_eq immediately (in 1 step)
    collide(f, rho, v, 1.0, Nx, Ny);
    Kokkos::fence();

    Kokkos::deep_copy(h_f, f);

    // f_eq_0 for rho = 1.0, v = (0,0) is w_0 = 4/9 (~0.44444)
    EXPECT_NEAR(h_f(2, 2, 0), 4.0 / 9.0, 1e-6);
    // f_eq_1 is w_1 = 1/9
    EXPECT_NEAR(h_f(2, 2, 1), 1.0 / 9.0, 1e-6);
}

// -----------------------------------------------------------------------------
// Test 3: streaming
// Ensures basic streaming moves distributions to neighboring cells appropriately.
// -----------------------------------------------------------------------------
TEST(LBMTest, Streaming) {
    int Nx = 5, Ny = 5;
    DistFuncD2Q9 f("f", Nx + 2, Ny + 2);
    DistFuncD2Q9 f_next("f_next", Nx + 2, Ny + 2);
    CellTypeField cell_type("cells", Nx + 2, Ny + 2);

    auto h_f = Kokkos::create_mirror_view(f);
    auto h_cell = Kokkos::create_mirror_view(cell_type);
    
    for (int y = 0; y < Ny + 2; ++y) {
        for (int x = 0; x < Nx + 2; ++x) {
            h_cell(x, y) = 0;
            for (int i = 0; i < 9; ++i) h_f(x, y, i) = 0.0;
        }
    }

    // Single particle probability pushing right (cx = 1, cy = 0 is i=1)
    h_f(2, 2, 1) = 1.0; 
    Kokkos::deep_copy(f, h_f);
    Kokkos::deep_copy(cell_type, h_cell);

    streaming(f, f_next, false, 0.0, 0, 0, Nx, Ny, cell_type, Nx, Ny);
    Kokkos::fence();

    auto h_f_next = Kokkos::create_mirror_view(f_next);
    Kokkos::deep_copy(h_f_next, f_next);

    // The particle should have transitioned cleanly from (2, 2) to (3, 2)
    EXPECT_DOUBLE_EQ(h_f_next(3, 2, 1), 1.0);
    EXPECT_DOUBLE_EQ(h_f_next(2, 2, 1), 0.0);
}

// -----------------------------------------------------------------------------
// Test 4: streaming (bounce-back)
// Validates that hitting a solid boundary reflects the particle to its opposite
// origin vector.
// -----------------------------------------------------------------------------
TEST(LBMTest, StreamingBounce) {
    int Nx = 5, Ny = 5;
    DistFuncD2Q9 f("f", Nx + 2, Ny + 2);
    DistFuncD2Q9 f_next("f_next", Nx + 2, Ny + 2);
    CellTypeField cell_type("cells", Nx + 2, Ny + 2);

    auto h_f = Kokkos::create_mirror_view(f);
    auto h_cell = Kokkos::create_mirror_view(cell_type);
    
    for (int y = 0; y < Ny + 2; ++y) {
        for (int x = 0; x < Nx + 2; ++x) {
            h_cell(x, y) = 0;
            for (int i = 0; i < 9; ++i) h_f(x, y, i) = 0.0;
        }
    }

    h_cell(3, 2) = 1;   // Deploy solid wall

    // Particle pushing right from (2, 2) attempting to breach the wall at (3, 2)
    h_f(2, 2, 1) = 1.0; 
    Kokkos::deep_copy(f, h_f);
    Kokkos::deep_copy(cell_type, h_cell);

    streaming(f, f_next, false, 0.0, 0, 0, Nx, Ny, cell_type, Nx, Ny);
    Kokkos::fence();

    auto h_f_next = Kokkos::create_mirror_view(f_next);
    Kokkos::deep_copy(h_f_next, f_next);

    // In a pull-scheme bounce-back:
    // f_out(2, 2, direction=3 [moving left]) looks at source cell (3, 2) [wall].
    // Reflects off (3, 2) taking opp[3]=1 from itself: f_in(2, 2, 1) = 1.0.
    EXPECT_DOUBLE_EQ(h_f_next(2, 2, 3), 1.0);
}

// -----------------------------------------------------------------------------
// Custom Entry Point
// Standard execution order for GoogleTests running through Kokkos and MPI hooks
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    
    int result = RUN_ALL_TESTS();
    
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}