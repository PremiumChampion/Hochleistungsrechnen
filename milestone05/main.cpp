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

    int Nx = 128;
    int Ny = 128;
    double u_lid = 0.1;
    double omega = 1.7;

    if (rank == 0) {
        std::cout << "Starting lid-driven cavity simulation (Re ~ 435)...\n";
    }

    simulation sim{Nx, Ny, omega, InitialisationPattern::Empty, true};
    sim.u_lid = u_lid;

    double max_diff = 1.0;
    double threshold = 1e-6;

    auto v_host = Kokkos::create_mirror_view(sim.v);
    auto v_prev = Kokkos::create_mirror_view(sim.v);

    Kokkos::deep_copy(v_prev, 0.0);

    int step = 0;
    int check_interval = 200;

    while (max_diff > threshold) {
        sim.step(check_interval - 1);
        Kokkos::fence();
        Kokkos::deep_copy(v_prev, sim.v);

        sim.step(1);
        Kokkos::fence();
        Kokkos::deep_copy(v_host, sim.v);

        double local_max_diff = 0.0;
        for (int y = 1; y <= sim.local_Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double dx = v_host(x, y, 0) - v_prev(x, y, 0);
                double dy = v_host(x, y, 1) - v_prev(x, y, 1);
                double diff = std::sqrt(dx * dx + dy * dy);
                if (diff > local_max_diff) {
                    local_max_diff = diff;
                }
            }
        }

        MPI_Allreduce(&local_max_diff, &max_diff, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);

        step += check_interval;
        if (rank == 0 && step % 2000 == 0) {
            std::cout << "Step " << step << " max velocity change: " << max_diff
                      << "\n";
        }
    }

    if (rank == 0)
        std::cout << "Converged to steady-state after " << step << " steps.\n";

    int bench_steps = 10000;
    Kokkos::fence();
    Kokkos::Timer timer;

    sim.step(bench_steps);
    Kokkos::fence();

    if (rank == 0) {
        double runtime = timer.seconds();
        double mlups = (static_cast<double>(Nx) * sim.global_Ny * bench_steps) /
                       (runtime * 1e6);
        std::cout << "\nRuntime over continuous " << bench_steps
                  << " steps: " << runtime << " s\n";
        std::cout << "Performance: " << mlups << " MLUPS\n\n";
    }

    auto global_v = sim.get_global_velocity();

    if (rank == 0) {
        std::ofstream out("lid_driven_cavity.csv");
        out << "x,y,ux,uy\n";
        for (int y = 0; y < sim.global_Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                out << x << "," << y << "," << global_v(x, y, 0) << ","
                    << global_v(x, y, 1) << "\n";
            }
        }
        std::cout << "Velocity field saved to lid_driven_cavity.csv\n";

        std::ofstream profile("centerline_profile.csv");
        profile << "y,ux\n";
        int center_x = Nx / 2;
        for (int y = 0; y < sim.global_Ny; ++y) {
            profile << y << "," << global_v(center_x, y, 0) << "\n";
        }
        std::cout
            << "Centerline velocity profile saved to centerline_profile.csv\n";
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}