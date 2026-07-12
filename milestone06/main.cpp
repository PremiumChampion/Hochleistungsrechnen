#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        int Nx = 1000;
        int Ny = 1000;
        double omega = 1.0;

        if (rank == 0) {
            std::cout << "Milestone 06: Distributed memory parallelization "
                         "with MPI\n";
            std::cout << "Grid: " << Nx << "x" << Ny << " distributed over "
                      << size << " ranks.\n";
            std::cout << "Starting warmup...\n";
        }

        // Creating simulation across active distributed memory ranks
        Simulation sim{Nx, Ny, omega, InitialisationPattern::Droplet, true};

        // Warm-up to bypass initial system overhead latency
            sim.step(100);
            Kokkos::fence();
            MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout
                << "Benchmarking Strong Scaling Performance (1000 steps)...\n";
        }

        int bench_steps = 1000;
        Kokkos::Timer timer;

        // Distributed computational effort over domains
        sim.step(bench_steps);
        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);

        double runtime = timer.seconds();

        // Globally synchronizing derived metrics using Collective Communication
        double total_mass = sim.get_total_density();
        double total_ke = sim.get_total_kinetic_energy();

        if (rank == 0) {
            // Calculate Megalattice Updates Per Second (MLUPS)
            double mlups =
                (static_cast<double>(Nx) * sim.global_Ny * bench_steps) /
                (runtime * 1e6);
            std::cout << "\n--- Final Results ---\n";
            std::cout << "Total mass: " << total_mass << "\n";
            std::cout << "Total kinetic energy: " << total_ke << "\n";
            std::cout << "Runtime: " << runtime << " s\n";
            std::cout << "Performance: " << mlups << " MLUPS\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}