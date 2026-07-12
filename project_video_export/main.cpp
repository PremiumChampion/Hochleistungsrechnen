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
        int Nx = 1920 / 2;
        int Ny = 1080 / 2;

        Simulation sim{Nx, Ny, 1};

        size_t render_after_steps = 10;
        size_t total_frames = 2000; // Total frames to render for the video

        FILE *ffmpeg = nullptr;

        // Only Rank 0 handles the video encoding
        if (rank == 0) {
            // Open a pipe to FFmpeg.
            // -y: overwrite output file if it exists
            // -f rawvideo: input format is raw RGB/RGBA data
            // -pix_fmt abgr: input pixel format matches our uint32_t structure
            // -s 600x400: resolution must match Nx and Ny
            // -r 60: output framerate (60 fps)
            // -i -: read input from stdin (the pipe)
            // -c:v libx264: use the H.264 video codec
            // -pix_fmt yuv420p: standard pixel format for compatibility (MP4
            // players) output.mp4: the output video file name
            std::string cmd =
                "ffmpeg -y -f rawvideo -pix_fmt abgr -s " + std::to_string(Nx) +
                "x" + std::to_string(sim.global_Ny) +
                " -r 60 -i - -c:v libx264 -pix_fmt yuv420p output.mp4";

            ffmpeg = popen(cmd.c_str(), "w");
            if (!ffmpeg) {
                std::cerr << "Error: Could not open pipe to FFmpeg. Make sure "
                             "FFmpeg is installed.\n";
            } else {
                std::cout << "Recording simulation to output.mp4...\n";
            }
        }

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Simulation and Recording Loop
        for (size_t frame = 0; frame < total_frames; ++frame) {
            sim.step(render_after_steps);
            auto h_pixels = sim.get_global_rgb_speed();

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
                std::cout << "\nVideo export complete! Saved to output.mp4\n";
            }
            std::cout << "Simulation and recording took: " << elapsed.count()
                      << " seconds.\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}