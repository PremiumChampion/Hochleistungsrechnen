#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int Nx = 128;
        int Ny = 128;
        double u_lid = 0.1;
        double omega = 1.7;

        std::cout << "Starting lid-driven cavity simulation (Re ~ 435)...\n";

        simulation sim{static_cast<uint>(Nx), static_cast<uint>(Ny), omega,
                       InitialisationPattern::Empty, true};
        sim.u_lid = u_lid;

        double max_diff = 1.0;
        double threshold = 1e-6; // Ensure precise steady state

        auto v_host = Kokkos::create_mirror_view(sim.v);
        auto v_prev = Kokkos::create_mirror_view(sim.v);

        Kokkos::deep_copy(v_prev, 0.0);

        int step = 0;
        int check_interval = 200;

        while (max_diff > threshold) {
            // Run bulk of the checks independently
            sim.step(check_interval - 1);
            Kokkos::fence();
            Kokkos::deep_copy(v_prev, sim.v); // Save state explicitly

            // Step singularly to find exact successive diff
            sim.step(1);
            Kokkos::fence();
            Kokkos::deep_copy(v_host, sim.v);

            max_diff = 0.0;
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    double dx = v_host(x, y, 0) - v_prev(x, y, 0);
                    double dy = v_host(x, y, 1) - v_prev(x, y, 1);
                    double diff = std::sqrt(dx * dx + dy * dy);
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
            }
            step += check_interval;
            if (step % 2000 == 0) {
                std::cout << "Step " << step
                          << " max velocity change: " << max_diff << "\n";
            }
        }

        std::cout << "Converged to steady-state after " << step << " steps.\n";

        // Benchmark strict performance explicitly without host communication
        int bench_steps = 10000;
        Kokkos::fence();
        Kokkos::Timer timer;

        sim.step(bench_steps);
        Kokkos::fence();

        double runtime = timer.seconds();
        double mlups =
            (static_cast<double>(Nx) * Ny * bench_steps) / (runtime * 1e6);
        std::cout << "\nRuntime over continuous " << bench_steps
                  << " steps: " << runtime << " s\n";
        std::cout << "Performance: " << mlups << " MLUPS\n\n";

        // Output steady state vector field
        std::ofstream out("lid_driven_cavity.csv");
        out << "x,y,ux,uy\n";
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                out << x << "," << y << "," << v_host(x, y, 0) << ","
                    << v_host(x, y, 1) << "\n";
            }
        }
        std::cout << "Velocity field saved to lid_driven_cavity.csv\n";

        // Output the centerline velocity profile
        std::ofstream profile("centerline_profile.csv");
        profile << "y,ux\n";
        int center_x = Nx / 2;
        for (int y = 0; y < Ny; ++y) {
            profile << y << "," << v_host(center_x, y, 0) << "\n";
        }
        std::cout
            << "Centerline velocity profile saved to centerline_profile.csv\n";
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}