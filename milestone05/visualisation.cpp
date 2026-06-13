#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

// Include SDL3
#include <SDL3/SDL.h>

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        // Setup Grid and Cavity Parameters
        uint Nx = 256;
        uint Ny = 256;
        double u_lid = 0.1;
        double omega = 1.7;

        if (rank == 0) {
            std::cout << "Starting Lid-Driven Cavity Live Demo...\n";
            std::cout << "Grid: " << Nx << "x" << Ny << "\n";
            std::cout << "Omega: " << omega << ", Lid Velocity: " << u_lid
                      << "\n";
            std::cout << "Close the window to exit.\n";
        }

        // Initialize simulation with Hard Walls (bounce = true) and Empty
        // distribution
        simulation sim{Nx, Ny, omega, InitialisationPattern::Empty, true};
        sim.u_lid = u_lid;

        // =========================================================================
        // SDL3 SETUP (Only on Rank 0)
        // =========================================================================
        SDL_Window *window = nullptr;
        SDL_Renderer *renderer = nullptr;
        SDL_Texture *texture = nullptr;

        if (rank == 0) {
            if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                std::cerr << "SDL Initialization failed: " << SDL_GetError()
                          << "\n";
                return 1;
            }

            // Scale the window up by a factor of 2 so it is easier to see
            int window_width = Nx * 2;
            int window_height = Ny * 2;
            window = SDL_CreateWindow("Lid-Driven Cavity Flow", window_width,
                                      window_height, 0);

            // Create a renderer and a texture matching the simulation grid
            // dimensions
            renderer = SDL_CreateRenderer(window, nullptr);
            texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                        SDL_TEXTUREACCESS_STREAMING, Nx, Ny);
        }

        bool quit = false;
        size_t render_after_steps = 15;

        // Run until the user closes the window
        while (!quit) {
            // Compute physics steps
            sim.step(render_after_steps);

            // Rank 0 handles the GUI and Events
            if (rank == 0) {
                SDL_Event e;
                while (SDL_PollEvent(&e)) {
                    if (e.type == SDL_EVENT_QUIT) {
                        quit = true;
                    }
                }

                // Copy only the computed color pixels from Device (GPU) to Host
                // (CPU)
                auto h_pixels = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, sim.rgb_direction);

                // Push pixels to the SDL texture and render
                SDL_UpdateTexture(texture, nullptr, h_pixels.data(),
                                  Nx * sizeof(uint32_t));
                SDL_RenderClear(renderer);
                SDL_RenderTexture(renderer, texture, nullptr, nullptr);
                SDL_RenderPresent(renderer);
            }

            // Sync the "quit" state to all other MPI processes so they exit
            // together
            int quit_int = quit ? 1 : 0;
            MPI_Bcast(&quit_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            quit = (quit_int == 1);
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