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
        // Widescreen 16:9 aspect ratio suitable for presentation slides
        int Nx = 1920;
        int Ny = 1080;

        // Omega = 1.75 keeps the simulation stable under pressure gradients
        double omega = 1.75;
        simulation sim{Nx, Ny, omega, InitialisationPattern::Empty, true};

        // Obstacle setup
        // Placed at 1/5th of the channel length
        int cx = Nx / 5;
        // Shifted by +3 units off-center to break vertical symmetry
        int cy = Ny / 2 + 3;
        int radius = Ny / 15; // Diameter of ~72 lattice nodes

        // Define internal boundaries and sources on the grid
        for (int y = 1; y <= sim.local_Ny; ++y) {
            int gy = y + sim.offset_y - 1;
            for (int x = 0; x < Nx; ++x) {
                // Default: fluid
                sim.h_cell_type(x, y) = 0;

                // Constant high-density source on the left boundary
                if (x == 1) {
                    sim.h_cell_type(x, y) = 2;
                }
                // Constant low-density sink on the right boundary
                else if (x == Nx - 2) {
                    sim.h_cell_type(x, y) = 3;
                }
                // Solid circular cylinder obstacle
                else {
                    double dx = x - cx;
                    double dy = gy - cy;
                    if (dx * dx + dy * dy <= radius * radius) {
                        sim.h_cell_type(x, y) = 1; // Wall
                    }
                }
            }
        }
        // Copy cell configuration from host mirror to active device memory
        Kokkos::deep_copy(sim.cell_type, sim.h_cell_type);

        size_t render_after_steps = 10;
        size_t total_frames = 1200; // Produces 20 seconds of video at 60 FPS

        FILE *ffmpeg = nullptr;

        // Rank 0 handles the FFmpeg encoding pipe
        if (rank == 0) {
            std::string cmd =
                "ffmpeg -y -f rawvideo -pix_fmt abgr -s " + std::to_string(Nx) +
                "x" + std::to_string(sim.global_Ny) +
                " -r 60 -i - -c:v libx264 -pix_fmt yuv420p output.mp4";

            ffmpeg = popen(cmd.c_str(), "w");
            if (!ffmpeg) {
                std::cerr
                    << "Error: Could not open pipe to FFmpeg. Confirm "
                       "FFmpeg is installed and accessible in the PATH.\n";
            } else {
                std::cout << "Starting wind-tunnel simulation. Exporting to "
                             "output.mp4...\n";
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Main Simulation and Video Generation Loop
        for (size_t frame = 0; frame < total_frames; ++frame) {
            sim.step(render_after_steps);

            // Retrieve direction-based color mapping for visual variety
            auto h_pixels = sim.get_global_rgb_direction();

            if (rank == 0 && ffmpeg) {
                // Overlay cylinder, source, and sink positions visually on Rank
                // 0
                for (int y = 0; y < Ny; ++y) {
                    for (int x = 0; x < Nx; ++x) {
                        double dx = x - cx;
                        double dy = y - cy;
                        if (dx * dx + dy * dy <= radius * radius) {
                            h_pixels(y * Nx + x) = 0x888888FF; // Grey Cylinder
                        } else if (x == 1) {
                            h_pixels(y * Nx + x) =
                                0x00FF00FF; // Green Source line
                        } else if (x == Nx - 2) {
                            h_pixels(y * Nx + x) = 0xFF0000FF; // Red Sink line
                        }
                    }
                }

                // Push raw frame into the encoder pipe
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

            // Simple error handling broadcast to exit if pipe fails
            int error = (ffmpeg == nullptr && rank == 0) ? 1 : 0;
            MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (error)
                break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        if (rank == 0) {
            if (ffmpeg) {
                pclose(ffmpeg);
                std::cout << "\nVideo export complete! Saved as output.mp4\n";
            }
            std::cout << "Simulation and recording took: " << elapsed.count()
                      << " seconds.\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}