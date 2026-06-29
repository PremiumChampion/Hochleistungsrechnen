#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        // Setup Grid and Cavity Parameters with a large canvas
        int Nx = 1024;
        int Ny = 1024;
        double u_lid = 0.1;
        double omega = 1.7;

        simulation sim{Nx, Ny, omega, InitialisationPattern::Empty, true};
        sim.u_lid = u_lid;

        size_t render_after_steps = 15;
        size_t total_frames = 2000;

        FILE *ffmpeg = nullptr;

        if (rank == 0) {
            std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt abgr -s " +
                              std::to_string(Nx) + "x" +
                              std::to_string(sim.global_Ny) +
                              " -r 60 -i - -c:v libx264 -pix_fmt yuv420p "
                              "milestone05_video.mp4";

            ffmpeg = popen(cmd.c_str(), "w");
            if (!ffmpeg) {
                std::cerr << "Error: Could not open pipe to FFmpeg. Make sure "
                             "FFmpeg is installed.\n";
            } else {
                std::cout
                    << "Recording simulation to milestone05_video.mp4...\n";
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t frame = 0; frame < total_frames; ++frame) {
            sim.step(render_after_steps);
            auto h_pixels = sim.get_global_rgb_direction();

            if (rank == 0 && ffmpeg) {
                // Write the raw pixel buffer directly to FFmpeg
                size_t written = fwrite(h_pixels.data(), sizeof(uint32_t),
                                        Nx * sim.global_Ny, ffmpeg);
                if (written != static_cast<size_t>(Nx * sim.global_Ny)) {
                    std::cerr << "Error writing frame to FFmpeg pipe!\n";
                }

                if (frame % 50 == 0) {
                    std::cout << "Rendered frame " << frame << " / "
                              << total_frames << "\n";
                }
            }

            // Sync MPI processes
            int error = (ffmpeg == nullptr && rank == 0) ? 1 : 0;
            MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (error)
                break;
        }

        // Stop timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        if (rank == 0) {
            if (ffmpeg) {
                pclose(ffmpeg);
                std::cout << "\nVideo export complete! Saved to "
                             "milestone05_video.mp4\n";
            }
            std::cout << "Simulation and recording took: " << elapsed.count()
                      << " seconds.\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}