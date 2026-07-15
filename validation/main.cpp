#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Run this validation strictly on 1 rank to easily probe the local host
    // memory
    if (size > 1) {
        if (rank == 0)
            std::cerr << "Error: Validation tests must be run with exactly 1 "
                         "MPI rank.\n";
        Kokkos::finalize();
        MPI_Finalize();
        return 1;
    }

    // -------------------------------------------------------------------------
    // TEST Global Mass Conservation & Relaxation (Acoustic Waves)
    // -------------------------------------------------------------------------
    {
        std::cout << "Global Mass Conservation and Density "
                     "Relaxation...\n";

        // Droplet pattern places a Gaussian density peak in the center
        Simulation sim{200, 200, 1.0, InitialisationPattern::Droplet, false};

        std::ofstream mass_file("mass_conservation.csv");
        mass_file << "step,total_mass\n";
        mass_file << std::setprecision(15);

        std::ofstream relax_file("relaxation.csv");
        relax_file << "t,x,rho\n";

        double initial_mass = sim.get_total_density();
        double max_mass_drift = 0.0;

        // Run for 200 time steps
        for (int t = 0; t <= 200; ++t) {

            // 1. Log Global Mass
            double current_mass = sim.get_total_density();
            max_mass_drift =
                std::max(max_mass_drift, std::abs(current_mass - initial_mass));
            mass_file << t << "," << current_mass << "\n";

            // 2. Log Centerline Density Profile (every 40 steps)
            if (t % 40 == 0) {
                auto h_rho = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), sim.rho);
                int y = sim.local_Ny / 2; // Extract middle slice
                for (int x = 1; x <= sim.local_Nx; ++x) {
                    relax_file << t << "," << x << "," << h_rho(x, y) << "\n";
                }
            }

            if (t < 200)
                sim.step(1);
        }

        std::cout << "  Data successfully saved to 'mass_conservation.csv' and "
                     "'relaxation.csv'.\n";
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}