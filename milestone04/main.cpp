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

    int Nx = 100;
    int Ny = 100;

    // 1. Plot evolution of velocity profile for omega = 1.0
    if (rank == 0)
        std::cout
            << "Simulating velocity profile evolution for omega = 1.0...\n";
    {
        Simulation sim{Nx, Ny, 1.0, InitialisationPattern::ShearWave, false};

        std::ofstream out;
        if (rank == 0) {
            out.open("velocity_profile.csv");
            out << "y,t,ux\n";
        }

        for (int t = 0; t <= 4000; t += 500) {
            auto global_v = sim.get_global_velocity();
            if (rank == 0) {
                for (int y = 0; y < Ny; ++y) {
                    out << y << "," << t << "," << global_v(0, y, 0) << "\n";
                }
            }
            if (t < 4000)
                sim.step(500);
        }
    }

    // 2. Measure analytical and computational viscosity
    if (rank == 0)
        std::cout << "Measuring viscosity vs omega...\n";

    // std::vector<double> omegas = {0.5, 0.6,
    // 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9};
    std::vector<double> omegas = {};
    for (double i = 0.2; i < 2.0; i += 0.05)
        omegas.push_back(i);

    std::ofstream out;
    if (rank == 0) {
        out.open("viscosity.csv");
        out << "omega,measured_nu,analytical_nu\n";
    }

    for (double omega : omegas) {
        Simulation sim{Nx, Ny, omega, InitialisationPattern::ShearWave, false};

        double k = 2.0 * M_PI / Ny;
        double analytical_nu = (1.0 / 3.0) * (1.0 / omega - 0.5);

        int steps = 1000;
        // sim.step(100);

        auto global_v = sim.get_global_velocity();
        double initial_amp = 0;
        if (rank == 0)
            initial_amp = global_v(0, Ny / 4, 0);

        sim.step(steps);

        global_v = sim.get_global_velocity();
        double final_amp = 0;
        if (rank == 0)
            final_amp = global_v(0, Ny / 4, 0);

        if (rank == 0) {
            double measured_nu =
                -std::log(final_amp / initial_amp) / (k * k * steps);
            out << omega << "," << measured_nu << "," << analytical_nu << "\n";
            std::cout << "Omega: " << omega << " | Measured Nu: " << measured_nu
                      << " | Analytical Nu: " << analytical_nu << "\n";
        }
    }

    if (rank == 0) {
        std::cout << "\nResults logged to "
                     "CSVs.\nvelocity_profile.csv\nviscosity.csv\n";
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}