#include "simulation_1d.h"
#include "simulation_2d.h"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>
#include <string>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    Kokkos::initialize(argc, argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        // Defaults
        int Nx = 2000;
        int Ny = 2000;
        int bench_steps = 1000;
        int dim = 2;
        double omega = 1.0;

        // Parse CLI arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--Nx" && i + 1 < argc)
                Nx = std::stoi(argv[++i]);
            else if (arg == "--Ny" && i + 1 < argc)
                Ny = std::stoi(argv[++i]);
            else if (arg == "--steps" && i + 1 < argc)
                bench_steps = std::stoi(argv[++i]);
            else if (arg == "--dim" && i + 1 < argc)
                dim = std::stoi(argv[++i]);
        }

        if (rank == 0) {
            std::cout << "Milestone 06: Advanced HPC Benchmarking\n";
            std::cout << "Grid: " << Nx << "x" << Ny << " distributed over "
                      << size << " ranks.\n";
            std::cout << "Decomposition: " << dim << "D\n";
            std::cout << "Starting warmup...\n";
        }

        // Shared variables for results
        double total_mass = 0, total_ke = 0;
        double compute_time = 0, comm_time = 0, runtime = 0;

        if (dim == 1) {
            Simulation1D sim{Nx, Ny, omega, InitialisationPattern::Droplet,
                             true};

            // Warm-up to bypass initial system overhead/JIT latency
            sim.step(100);
            Kokkos::fence();
            MPI_Barrier(MPI_COMM_WORLD);
            sim.total_compute_time = 0.0;
            sim.total_comm_time = 0.0;

            if (rank == 0)
                std::cout << "Benchmarking " << bench_steps << " steps...\n";

            Kokkos::Timer timer;
            sim.step(bench_steps);

            Kokkos::fence();
            MPI_Barrier(MPI_COMM_WORLD);
            runtime = timer.seconds();

            total_mass = sim.get_total_density();
            total_ke = sim.get_total_kinetic_energy();
            compute_time = sim.total_compute_time;
            comm_time = sim.total_comm_time;

        } else {
            Simulation2D sim{Nx, Ny, omega, InitialisationPattern::Droplet,
                             true};

            sim.step(100);
            Kokkos::fence();
            MPI_Barrier(MPI_COMM_WORLD);
            sim.total_compute_time = 0.0;
            sim.total_comm_time = 0.0;

            if (rank == 0)
                std::cout << "Benchmarking " << bench_steps << " steps...\n";

            Kokkos::Timer timer;
            sim.step(bench_steps);

            Kokkos::fence();
            MPI_Barrier(MPI_COMM_WORLD);
            runtime = timer.seconds();

            total_mass = sim.get_total_density();
            total_ke = sim.get_total_kinetic_energy();
            compute_time = sim.total_compute_time;
            comm_time = sim.total_comm_time;
        }

        if (rank == 0) {
            // Calculate Performance Metrics
            double mlups =
                (static_cast<double>(Nx) * Ny * bench_steps) / (runtime * 1e6);
            double compute_percent = (compute_time / runtime) * 100.0;
            double comm_percent = (comm_time / runtime) * 100.0;

            std::cout << "\n--- Final Results ---\n";
            std::cout << "Total mass: " << total_mass << "\n";
            std::cout << "Total kinetic energy: " << total_ke << "\n";
            std::cout << "Runtime: " << runtime << " s\n";
            std::cout << "Calculation Time (GPU): " << compute_time << " s ("
                      << compute_percent << "%)\n";
            std::cout << "Networking Time (MPI): " << comm_time << " s ("
                      << comm_percent << "%)\n";
            std::cout << "Performance: " << mlups << " MLUPS\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}