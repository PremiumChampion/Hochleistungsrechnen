// #include "simulation_1d.h"
// #include "simulation_2d.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
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
        std::string scaling_type = "strong"; // "strong" or "weak"
        std::string csv_file = "benchmark_results.csv";

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
            else if (arg == "--omega" && i + 1 < argc)
                omega = std::stod(argv[++i]);
            else if (arg == "--scaling" && i + 1 < argc)
                scaling_type = argv[++i];
            else if (arg == "--csv" && i + 1 < argc)
                csv_file = argv[++i];
        }

        if (rank == 0) {
            std::cout << "=== LBM Benchmark ===\n";
            std::cout << "Grid: " << Nx << "x" << Ny << " | Tasks: " << size
                      << " | Dim: " << dim << "D"
                      << " | Steps: " << bench_steps
                      << " | Scaling: " << scaling_type << "\n";
            std::cout << "Starting warmup (100 steps)...\n";
        }

        // Shared variables for results
        double total_mass = 0, total_ke = 0;
        double compute_time = 0, comm_time = 0, runtime = 0;

        Simulation sim{
            Nx,    Ny,
            omega, InitialisationPattern::Droplet,
            true,  dim == 1 ? DecompType::DECOMP_1D : DecompType::DECOMP_2D};

        // Warm-up to bypass JIT latency and cache warming
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

        if (rank == 0) {
            // Calculate Performance Metrics
            double mlups =
                (static_cast<double>(Nx) * Ny * bench_steps) / (runtime * 1e6);
            double compute_percent = (compute_time / runtime) * 100.0;
            double comm_percent = (comm_time / runtime) * 100.0;
            double other_percent = 100.0 - compute_percent - comm_percent;

            // Strong scaling efficiency (relative to 1 task)
            // This will be computed in post-processing from CSV
            double cells_per_task = static_cast<double>(Nx) * Ny / size;
            double mlups_per_task = mlups / size;

            std::cout << "\n--- Final Results ---\n";
            std::cout << "Total mass: " << total_mass << "\n";
            std::cout << "Total kinetic energy: " << total_ke << "\n";
            std::cout << "Runtime: " << runtime << " s\n";
            std::cout << "Calculation Time (GPU): " << compute_time << " s ("
                      << compute_percent << "%)\n";
            std::cout << "Networking Time (MPI): " << comm_time << " s ("
                      << comm_percent << "%)\n";
            std::cout << "Other overhead: "
                      << (runtime - compute_time - comm_time) << " s ("
                      << other_percent << "%)\n";
            std::cout << "Performance: " << mlups << " MLUPS\n";
            std::cout << "Performance per task: " << mlups_per_task
                      << " MLUPS/task\n";
            std::cout << "Cells per task: " << cells_per_task << "\n";

            // Write CSV (append mode)
            std::ofstream outfile;
            bool file_exists = std::ifstream(csv_file).good();
            outfile.open(csv_file, std::ios::app);

            if (!file_exists) {
                outfile << "scaling_type,dim,Nx,Ny,tasks,steps,omega,"
                        << "runtime,compute_time,comm_time,other_time,"
                        << "compute_pct,comm_pct,other_pct,"
                        << "mlups,mlups_per_task,cells_per_task,"
                        << "total_mass,total_ke\n";
            }

            outfile << scaling_type << "," << dim << "," << Nx << "," << Ny
                    << "," << size << "," << bench_steps << "," << omega << ","
                    << runtime << "," << compute_time << "," << comm_time << ","
                    << (runtime - compute_time - comm_time) << ","
                    << compute_percent << "," << comm_percent << ","
                    << other_percent << "," << mlups << "," << mlups_per_task
                    << "," << cells_per_task << "," << total_mass << ","
                    << total_ke << "\n";
            outfile.close();

            std::cout << "\nResults appended to " << csv_file << "\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}