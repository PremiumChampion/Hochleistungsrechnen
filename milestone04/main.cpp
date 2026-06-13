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

    // Limit evaluation metrics strictly to one Rank
    if (rank == 0) {
        int Nx = 100;
        int Ny = 100;

        // 1. Plot evolution of velocity profile for omega = 1.0 (approaching
        // limit t -> infinity)
        {
            std::cout
                << "Simulating velocity profile evolution for omega = 1.0...\n";
            // Initialize ShearWave with `bounce = false` so Periodic boundary
            // conditions are active.
            simulation sim{static_cast<uint>(Nx), static_cast<uint>(Ny), 1.0,
                           InitialisationPattern::ShearWave, false};

            std::ofstream out("velocity_profile.csv");
            out << "y,t,ux\n";

            auto v_host = Kokkos::create_mirror_view(sim.v);

            // Advance in steps of 500 up to 4000 to clearly observe complete
            // decay (t -> infinity).
            for (int t = 0; t <= 4000; t += 500) {
                Kokkos::deep_copy(v_host, sim.v);
                for (int y = 0; y < Ny; ++y) {
                    out << y << "," << t << "," << v_host(0, y, 0) << "\n";
                }
                if (t < 4000)
                    sim.step(500);
            }
        }

        // 2. Measure analytical and computational viscosity across a spread of
        // Omega ranges.
        std::cout << "Measuring viscosity vs omega...\n";
        std::vector<double> omegas = {0.5, 0.6, 0.8, 1.0, 1.2,
                                      1.4, 1.6, 1.8, 1.9};

        std::ofstream out("viscosity.csv");
        out << "omega,measured_nu,analytical_nu\n";

        for (double omega : omegas) {
            simulation sim{static_cast<uint>(Nx), static_cast<uint>(Ny), omega,
                           InitialisationPattern::ShearWave, false};

            double k = 2.0 * M_PI / Ny;
            double analytical_nu = (1.0 / 3.0) * (1.0 / omega - 0.5);

            int steps = 1000;
            auto v_host = Kokkos::create_mirror_view(sim.v);

            // Initializing $f_i$ directly to equilibrium without Chapman-Enskog
            // modifications yields minor acoustic artifacts. Therefore we
            // stabilize the wave by progressing 100 frames before reading
            // Initial Amp.
            sim.step(100);
            Kokkos::deep_copy(v_host, sim.v);

            // The sine wave has a peak velocity amplitude at Ny/4. (Since
            // Nx=Ny=100 => peak forms exactly at index 25)
            double initial_amp = v_host(0, Ny / 4, 0);

            // Simulate progression
            sim.step(steps);
            Kokkos::deep_copy(v_host, sim.v);
            double final_amp = v_host(0, Ny / 4, 0);

            // Calculate Kinematic Viscosity Nu derived from decay amplitude
            // formulas
            double measured_nu =
                -std::log(final_amp / initial_amp) / (k * k * steps);

            out << omega << "," << measured_nu << "," << analytical_nu << "\n";
            std::cout << "Omega: " << omega << " | Measured Nu: " << measured_nu
                      << " | Analytical Nu: " << analytical_nu << "\n";
        }
        std::cout << "\nResults logged to "
                     "CSVs.\nvelocity_profile.csv\nviscosity.csv\n";
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}