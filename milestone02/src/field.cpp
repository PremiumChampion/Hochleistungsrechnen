#include <Kokkos_Core.hpp>
#include <cmath>

// Define alias types for cleaner code
// We use compile-time dimensions ([2] and [9]) for the last indices.
using ScalarField2D = Kokkos::View<double **>;
using VectorField2D = Kokkos::View<double **[2]>;
using DistFuncD2Q9 = Kokkos::View<double **[9]>;

#define INIT_PATTERN_EMPTY 0
#define INIT_PATTERN_WAVE 1
#define INIT_PATTERN_SINGE 2
#define INIT_PATTERN INIT_PATTERN_WAVE

void initialize_lbm(int Nx, int Ny, ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next) {
    // 1. Density: size (Nx, Ny)
    rho = ScalarField2D("density", Nx, Ny);

    // 2. Velocity: size (Nx, Ny, 2)
    v = VectorField2D("velocity", Nx, Ny);

    // 3. Distribution function: size (Nx, Ny, 9)
    f = DistFuncD2Q9("distribution_function", Nx, Ny);

    // (Optional) You usually also need a second view for the distribution
    // function to store the updated values during the streaming step (f_new or
    // f_post_collision).
    f_next = DistFuncD2Q9("distribution_function_next", Nx, Ny);

    // Initialise the Views inside a Kokkos parallel execution:
#if INIT_PATTERN == INIT_PATTERN_EMPTY
    Kokkos::parallel_for(
        "InitFields", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // Initialize density to 1.0
            rho(x, y) = 1.0;

            // Initialize velocity components to 0.0
            v(x, y, 0) = 0.0; // u_x
            v(x, y, 1) = 0.0; // u_y

            for (int i = 0; i < 9; ++i) {
                f(x, y, i) = 0.0;
            }
        });
#endif
#if INIT_PATTERN == INIT_PATTERN_WAVE
    // Initialise the Views inside a Kokkos parallel execution:
    Kokkos::parallel_for(
        "InitFieldsWave",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // D2Q9 Parameters
            const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,  1. / 9., 1. / 9.,
                                 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
            const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                  1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                  1.0, 1.0, -1.0, -1.0};

            // Wave properties
            const double PI = 3.14159265358979323846;
            // One full wave across the X-achsis
            const double wave_length = Nx;
            const double k = 2.0 * PI / wave_length; // Wavenumber

            // Set amplitude for the wave (keep it small, Mach < 0.1 for
            //   LBM
            // stability)
            const double amp = 0.05;

            // Define the sinusoidal macroscopic fields
            // The wave travels along the X-axis. Y-velocity remains 0.
            double current_rho = 1.0 + amp * Kokkos::sin(k * x);
            double vx = amp * Kokkos::sin(k * x);
            double vy = 0.01;

            // Store macroscopic variables
            rho(x, y) = current_rho;
            v(x, y, 0) = vx;
            v(x, y, 1) = vy;

            // Initialize the distribution function to the equilibrium
            double u_sq = vx * vx + vy * vy;

            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * vx + cy[i] * vy;

                // D2Q9 Equilibrium Function Formula incorporating the
                // local perturbed density
                f(x, y, i) = w[i] * current_rho *
                             (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            }
        });
#endif
#if INIT_PATTERN == INIT_PATTERN_SINGE
    Kokkos::parallel_for(
        "InitFieldsSimple",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = 0.0;
            v(x, y, 0) = 0.0;
            v(x, y, 1) = 0.0;

            // Set all directions to 0 initially
            for (int i = 0; i < 9; ++i) {
                f(x, y, i) = 0.0;
            }

            // Place a single cluster of particles in the middle of the grid,
            // moving exclusively to the right (direction index 1, where cx = 1,
            // cy = 0)
            if (x == Nx / 2 && y == Ny / 2) {
                f(x, y, 1) = 1.0;
            }
        });
#endif
}

void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f) {
    // Extract grid dimensions directly from the View extents
    const int Nx = rho.extent(0);
    const int Ny = rho.extent(1);

    // Launch a 2D parallel loop over all spatial nodes (x, y)
    Kokkos::parallel_for(
        "ComputeMacroscopicFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // Standard D2Q9 discrete velocity components
            // Note: Storing them inside the lambda ensures they are available
            // on the GPU device
            const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                  1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                  1.0, 1.0, -1.0, -1.0};

            double local_rho = 0.0;
            double local_vx = 0.0;
            double local_vy = 0.0;

            // Compute the integrals (as discrete sums)
            for (int i = 0; i < 9; ++i) {
                double f_i = f(x, y, i);

                local_rho += f_i;
                local_vx += f_i * cx[i];
                local_vy += f_i * cy[i];
            }

            // Store the computed density
            rho(x, y) = local_rho;

            // Store the computed velocity components (v = sum / rho)
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out) {
    // Extract grid dimensions from the View
    const int Nx = f_out.extent(0);
    const int Ny = f_out.extent(1);

    // 2D parallel loop over all spatial nodes (x, y)
    Kokkos::parallel_for(
        "StreamingStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // Standard D2Q9 discrete velocity components (integers for grid
            // indexing)
            const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            for (int i = 0; i < 9; ++i) {
                // Determine the coordinates of the neighboring node to "pull"
                // from. We subtract cx[i] and cy[i] because we are looking
                // BACKWARDS to the source node that sent the particle towards
                // (x, y).

                // Adding Nx and Ny before the modulo operator ensures that
                // the dividend is positive, properly handling wrap-around for
                // periodic boundary conditions on the left/bottom edges.
                int src_x = (x - cx[i] + Nx) % Nx;
                int src_y = (y - cy[i] + Ny) % Ny;

                // Copy the distribution function from the source node
                f_out(x, y, i) = f_in(src_x, src_y, i);
            }
        });
}