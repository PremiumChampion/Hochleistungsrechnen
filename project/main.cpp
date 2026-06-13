#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <vector>

// Include SDL3
#include <SDL3/SDL.h>
#include <cstdint>

// Helper function to map velocity to an RGBA color
uint32_t velocity_to_color(double vx, double vy, double max_vel = 0.1) {
    double speed = std::sqrt(vx * vx + vy * vy);
    double intensity = std::min(speed / max_vel, 1.0);

    // Heatmap gradient: Blue (slow) to Red (fast)
    uint8_t r = static_cast<uint8_t>(intensity * 255);
    uint8_t g = 0;
    uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255);
    uint8_t a = 255;

    // SDL_PIXELFORMAT_RGBA8888 standard bit-shifting
    return (r << 24) | (g << 16) | (b << 8) | a;
}

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        std::cout << "Hello I am rank " << rank << " of " << size << "\n";

        uint Nx = 150 * 4;
        uint Ny = 100 * 4;
        simulation sim{Nx, Ny, 0.1};

        // =========================================================================
        // SDL3 SETUP (Only on Rank 0)
        // =========================================================================
        SDL_Window *window = nullptr;
        SDL_Renderer *renderer = nullptr;
        SDL_Texture *texture = nullptr;
        // std::vector<uint32_t> pixels(Nx * Ny);

        if (rank == 0) {
            if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                std::cerr << "SDL Initialization failed: " << SDL_GetError()
                          << "\n";
                return 1;
            }

            // Make the window larger than the grid so it's easy to see
            int window_width = 600;
            int window_height = 400;
            window = SDL_CreateWindow("LBM Velocity Magnitude", window_width,
                                      window_height, 0);

            // Create a renderer (nullptr picks the best default engine)
            renderer = SDL_CreateRenderer(window, nullptr);

            // Create a texture matching the simulation grid dimensions
            texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                        SDL_TEXTUREACCESS_STREAMING, Nx, Ny);
        }

        bool quit = false;
        int step = 0;

        size_t render_after_steps = 10;
        // Run until the user closes the window
        while (!quit) {

            // do the simulation
            sim.step(render_after_steps);

            // Rank 0 handles the GUI and Events
            if (rank == 0) {
                SDL_Event e;
                while (SDL_PollEvent(&e)) {
                    if (e.type == SDL_EVENT_QUIT) {
                        quit = true;
                    }
                }

                // Copy only the computed pixels from Device to Host
                auto h_pixels = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, sim.rgb_speed);

                // Push pixels to the SDL texture and render
                SDL_UpdateTexture(texture, nullptr, h_pixels.data(),
                                  Nx * sizeof(uint32_t));
                SDL_RenderClear(renderer);
                SDL_RenderTexture(renderer, texture, nullptr, nullptr);
                SDL_RenderPresent(renderer);
            }

            // Sync the "quit" state to all other MPI processes so they don't
            // hang
            int quit_int = quit ? 1 : 0;
            MPI_Bcast(&quit_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            quit = (quit_int == 1);
            step++;
        }

        if (rank == 0) {
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}