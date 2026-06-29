#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

// Include SDL3
#include <SDL3/SDL.h>

void serialize(const simulation &sim, const std::string &filename) {
    auto rho_h = Kokkos::create_mirror_view(sim.rho);
    auto v_h = Kokkos::create_mirror_view(sim.v);
    auto f_h = Kokkos::create_mirror_view(sim.f);

    Kokkos::deep_copy(rho_h, sim.rho);
    Kokkos::deep_copy(v_h, sim.v);
    Kokkos::deep_copy(f_h, sim.f);

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for writing.\n";
        return;
    }

    int Nx = sim.global_Nx;
    int Ny = sim.global_Ny;
    double omega = sim.omega;

    out.write(reinterpret_cast<const char *>(&Nx), sizeof(int));
    out.write(reinterpret_cast<const char *>(&Ny), sizeof(int));
    out.write(reinterpret_cast<const char *>(&omega), sizeof(double));

    out.write(reinterpret_cast<const char *>(rho_h.data()),
              rho_h.size() * sizeof(double));
    out.write(reinterpret_cast<const char *>(v_h.data()),
              v_h.size() * sizeof(double));
    out.write(reinterpret_cast<const char *>(f_h.data()),
              f_h.size() * sizeof(double));

    std::cout << "Simulation serialized to " << filename << "\n";
}

void deserialize(simulation &sim, const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for reading.\n";
        return;
    }

    int Nx, Ny;
    double omega;

    in.read(reinterpret_cast<char *>(&Nx), sizeof(int));
    in.read(reinterpret_cast<char *>(&Ny), sizeof(int));
    in.read(reinterpret_cast<char *>(&omega), sizeof(double));

    sim = simulation(Nx, Ny, omega);

    auto rho_h = Kokkos::create_mirror_view(sim.rho);
    auto v_h = Kokkos::create_mirror_view(sim.v);
    auto f_h = Kokkos::create_mirror_view(sim.f);

    in.read(reinterpret_cast<char *>(rho_h.data()),
            rho_h.size() * sizeof(double));
    in.read(reinterpret_cast<char *>(v_h.data()), v_h.size() * sizeof(double));
    in.read(reinterpret_cast<char *>(f_h.data()), f_h.size() * sizeof(double));

    Kokkos::deep_copy(sim.rho, rho_h);
    Kokkos::deep_copy(sim.v, v_h);
    Kokkos::deep_copy(sim.f, f_h);

    std::cout << "Simulation deserialized from " << filename << "\n";
}

int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        if (argc < 3) {
            if (rank == 0) {
                std::cout << "Usage: ./serial <save|load> <filename> "
                             "[steps_for_save]\n";
                std::cout << "Example save: ./serial save sim_state.bin 100\n";
                std::cout << "Example load: ./serial load sim_state.bin\n";
            }
        } else {
            std::string mode = argv[1];
            std::string filename = argv[2];

            if (mode == "save") {
                if (rank == 0)
                    std::cout << "Running simulation and saving to " << filename
                              << "...\n";

                uint Nx = 150 * 4;
                uint Ny = 100 * 4;
                simulation sim{static_cast<int>(Nx), static_cast<int>(Ny), 1.0};

                size_t steps = 100;
                if (argc > 3)
                    steps = std::stoul(argv[3]);

                sim.step(steps);
                serialize(sim, filename);

                if (rank == 0)
                    std::cout << "State saved successfully.\n";

            } else if (mode == "load") {
                if (rank == 0)
                    std::cout << "Loading simulation from " << filename
                              << " and visualizing...\n";

                simulation sim;
                deserialize(sim, filename);

                int Nx = sim.global_Nx;
                int Ny = sim.global_Ny;

                SDL_Window *window = nullptr;
                SDL_Renderer *renderer = nullptr;
                SDL_Texture *texture = nullptr;

                if (rank == 0) {
                    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                        std::cerr
                            << "SDL Initialization failed: " << SDL_GetError()
                            << "\n";
                    } else {
                        int window_width = 600;
                        int window_height = 400;
                        window =
                            SDL_CreateWindow("LBM Deserialized Visualization",
                                             window_width, window_height, 0);
                        renderer = SDL_CreateRenderer(window, nullptr);
                        texture = SDL_CreateTexture(
                            renderer, SDL_PIXELFORMAT_RGBA8888,
                            SDL_TEXTUREACCESS_STREAMING, Nx, Ny);
                    }
                }

                bool quit = false;

                while (!quit) {
                    sim.step(10);

                    auto pixels = sim.get_global_rgb_speed();

                    if (rank == 0) {
                        SDL_Event e;
                        while (SDL_PollEvent(&e)) {
                            if (e.type == SDL_EVENT_QUIT)
                                quit = true;
                        }

                        if (texture && renderer) {
                            SDL_UpdateTexture(texture, nullptr, pixels.data(),
                                              Nx * sizeof(uint32_t));
                            SDL_RenderClear(renderer);
                            SDL_RenderTexture(renderer, texture, nullptr,
                                              nullptr);
                            SDL_RenderPresent(renderer);
                        }
                    }

                    int quit_int = quit ? 1 : 0;
                    MPI_Bcast(&quit_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    quit = (quit_int == 1);
                }

                if (rank == 0) {
                    if (texture)
                        SDL_DestroyTexture(texture);
                    if (renderer)
                        SDL_DestroyRenderer(renderer);
                    if (window)
                        SDL_DestroyWindow(window);
                    SDL_Quit();
                }
            } else {
                if (rank == 0)
                    std::cerr << "Unknown mode: " << mode
                              << ". Use 'save' or 'load'.\n";
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}