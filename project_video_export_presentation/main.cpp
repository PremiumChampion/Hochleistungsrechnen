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
        // Full HD 1080p Resolution
        int Nx = 1920;
        int Ny = 1080;

        // Omega = 1.75 balances low viscosity (for high turbulence) with
        // numerical stability
        double omega = 1.7;
        simulation sim{Nx, Ny, omega, InitialisationPattern::Empty, true};

        // --- SCENE GEOMETRY DEFINITION ---
        // A helper lambda to easily define our complex obstacles using global
        // coordinates
        auto is_wall = [](int x, int y) -> bool {
            // 1. The Breakwater (Large central cylinder)
            if ((x - 350) * (x - 350) + (y - 540) * (y - 540) <= 120 * 120)
                return true;

            // 2. The Twin Scatterers (Two smaller cylinders)
            if ((x - 750) * (x - 750) + (y - 250) * (y - 250) <= 90 * 90)
                return true;
            if ((x - 750) * (x - 750) + (y - 830) * (y - 830) <= 90 * 90)
                return true;

            // 3. The Collector (Massive secondary cylinder offset slightly to
            // break symmetry)
            if ((x - 1200) * (x - 1200) + (y - 560) * (y - 560) <= 140 * 140)
                return true;

            // 4. The Chicane (Staggered vertical baffles)
            // Bottom wall upwards
            if (x >= 1550 && x <= 1600 && y >= 0 && y <= 450)
                return true;
            // Top wall downwards
            if (x >= 1700 && x <= 1750 && y >= 630 && y <= 1080)
                return true;

            return false;
        };

        // Map the geometry onto the distributed MPI grid
        for (int y = 1; y <= sim.local_Ny; ++y) {
            int gy = y + sim.offset_y - 1; // Global Y coordinate
            for (int x = 0; x < Nx; ++x) {
                sim.h_cell_type(x, y) = 0; // Default: fluid
                if (y % 4 == 0)
                    if (x < 10) {
                        sim.h_cell_type(x, y) = 2; // Source zone (Inlet)
                    } else if (x > Nx - 10) {
                        sim.h_cell_type(x, y) = 3; // Sink zone (Outlet)
                    } else if (is_wall(x, gy)) {
                        sim.h_cell_type(x, y) = 1; // Solid Obstacle
                    }
            }
        }
        // Push the configuration to device memory
        Kokkos::deep_copy(sim.cell_type, sim.h_cell_type);

        // Render settings
        size_t render_after_steps = 20;
        size_t total_frames = 20 * 60; // 30 seconds of video at 60 FPS

        // skip the first 5 seconds
        // sim.step(render_after_steps * 5 * 60);

        // Define 3 separate pipes for the 3 videos
        FILE *ffmpeg_dir = nullptr;
        FILE *ffmpeg_den = nullptr;
        FILE *ffmpeg_spd = nullptr;

        // Rank 0 handles the FFmpeg encoding pipes
        if (rank == 0) {
            std::string cmd_base =
                "ffmpeg -y -f rawvideo -pix_fmt abgr -s " + std::to_string(Nx) +
                "x" + std::to_string(Ny) +
                " -r 60 -i - -c:v libx264 -preset fast -pix_fmt "
                "yuv420p ";

            ffmpeg_dir =
                popen((cmd_base + "output_1080p_direction.mp4").c_str(), "w");
            ffmpeg_den =
                popen((cmd_base + "output_1080p_density.mp4").c_str(), "w");
            ffmpeg_spd =
                popen((cmd_base + "output_1080p_speed.mp4").c_str(), "w");

            if (!ffmpeg_dir || !ffmpeg_den || !ffmpeg_spd) {
                std::cerr
                    << "Error: Could not open one or more pipes to FFmpeg.\n";
            } else {
                std::cout << "Starting 1080p Obstacle Course simulation...\n";
                std::cout
                    << "Exporting 30 seconds of video to "
                       "output_1080p_direction.mp4, output_1080p_density.mp4, "
                       "and output_1080p_speed.mp4\n";
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Main Simulation and Video Generation Loop
        for (size_t frame = 0; frame < total_frames; ++frame) {

            // Advance the simulation
            sim.step(render_after_steps);

            // Fetch the visualisations
            auto h_pixels_dir = sim.get_global_rgb_direction();
            auto h_pixels_den = sim.get_global_rgb_density();
            auto h_pixels_spd = sim.get_global_rgb_speed();

            if (rank == 0 && ffmpeg_dir && ffmpeg_den && ffmpeg_spd) {
                // Overlay the solid geometry and boundaries onto the video
                // frames
                for (int y = 0; y < Ny; ++y) {
                    for (int x = 0; x < Nx; ++x) {
                        uint32_t overlay_color = 0;
                        bool apply_overlay = false;

                        if (is_wall(x, y)) {
                            // Dark gray color for obstacles (0xRRGGBBAA format)
                            overlay_color = 0x333333FF;
                            apply_overlay = true;
                        } else if (x < 10) {
                            // Green Source line
                            overlay_color = 0x00AA00FF;
                            apply_overlay = true;
                        } else if (x > Nx - 10) {
                            // Red Sink line
                            overlay_color = 0xAA0000FF;
                            apply_overlay = true;
                        }

                        if (apply_overlay) {
                            size_t idx = y * Nx + x;
                            h_pixels_dir(idx) = overlay_color;
                            h_pixels_den(idx) = overlay_color;
                            h_pixels_spd(idx) = overlay_color;
                        }
                    }
                }

                // Push raw frames into the respective encoder pipes
                size_t written_dir = fwrite(
                    h_pixels_dir.data(), sizeof(uint32_t), Nx * Ny, ffmpeg_dir);
                size_t written_den = fwrite(
                    h_pixels_den.data(), sizeof(uint32_t), Nx * Ny, ffmpeg_den);
                size_t written_spd = fwrite(
                    h_pixels_spd.data(), sizeof(uint32_t), Nx * Ny, ffmpeg_spd);

                if (written_dir != static_cast<size_t>(Nx * Ny) ||
                    written_den != static_cast<size_t>(Nx * Ny) ||
                    written_spd != static_cast<size_t>(Nx * Ny)) {
                    std::cerr << "Error writing frames to FFmpeg pipes!\n";
                }

                if (frame % 50 == 0) {
                    std::cout << "Rendered frame " << frame << " / "
                              << total_frames << "\n";
                }
            }

            // Simple error handling broadcast to exit cleanly if any pipe fails
            int error =
                ((!ffmpeg_dir || !ffmpeg_den || !ffmpeg_spd) && rank == 0) ? 1
                                                                           : 0;
            MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (error)
                break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        if (rank == 0) {
            if (ffmpeg_dir)
                pclose(ffmpeg_dir);
            if (ffmpeg_den)
                pclose(ffmpeg_den);
            if (ffmpeg_spd)
                pclose(ffmpeg_spd);

            std::cout << "\nVideo export complete!\n"
                      << "Saved as:\n"
                      << "- output_1080p_direction.mp4\n"
                      << "- output_1080p_density.mp4\n"
                      << "- output_1080p_speed.mp4\n";
            std::cout << "Simulation and recording took: " << elapsed.count()
                      << " seconds.\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}