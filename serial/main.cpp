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
    // Create host mirrors to copy data from the device (GPU) to the host (CPU)
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

    int Nx = sim.Nx;
    int Ny = sim.Ny;
    double omega = sim.omega;

    // Write header (dimensions and parameter)
    out.write(reinterpret_cast<const char *>(&Nx), sizeof(int));
    out.write(reinterpret_cast<const char *>(&Ny), sizeof(int));
    out.write(reinterpret_cast<const char *>(&omega), sizeof(double));

    // Write raw view data
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

    // Read header
    in.read(reinterpret_cast<char *>(&Nx), sizeof(int));
    in.read(reinterpret_cast<char *>(&Ny), sizeof(int));
    in.read(reinterpret_cast<char *>(&omega), sizeof(double));

    // Initialize the simulation with the loaded dimensions and omega
    // This allocates the necessary Kokkos views (including f_next)
    sim = simulation(Nx, Ny, omega);

    auto rho_h = Kokkos::create_mirror_view(sim.rho);
    auto v_h = Kokkos::create_mirror_view(sim.v);
    auto f_h = Kokkos::create_mirror_view(sim.f);

    // Read raw view data
    in.read(reinterpret_cast<char *>(rho_h.data()),
            rho_h.size() * sizeof(double));
    in.read(reinterpret_cast<char *>(v_h.data()), v_h.size() * sizeof(double));
    in.read(reinterpret_cast<char *>(f_h.data()), f_h.size() * sizeof(double));

    // Copy data from host back to the device
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
                std::cout << "Usage: ./project_serial <save|load> <filename> "
                             "[steps_for_save]\n";
                std::cout << "Example save: ./project_serial save "
                             "sim_state.bin 100\n";
                std::cout
                    << "Example load: ./project_serial load sim_state.bin\n";
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
                simulation sim{Nx, Ny, 1.0};

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

                int Nx = sim.Nx;
                int Ny = sim.Ny;

                // SDL3 SETUP (Only on Rank 0)
                SDL_Window *window = nullptr;
                SDL_Renderer *renderer = nullptr;
                SDL_Texture *texture = nullptr;
                std::vector<uint32_t> pixels(Nx * Ny);

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

                auto v_host = Kokkos::create_mirror_view(sim.v);
                bool quit = false;

                while (!quit) {
                    sim.step(10); // Continue simulation from deserialized state

                    if (rank == 0) {
                        SDL_Event e;
                        while (SDL_PollEvent(&e)) {
                            if (e.type == SDL_EVENT_QUIT) {
                                quit = true;
                            }
                        }

                        Kokkos::deep_copy(v_host, sim.v);

                        for (int y = 0; y < Ny; ++y) {
                            for (int x = 0; x < Nx; ++x) {
                                double vx = v_host(x, y, 0);
                                double vy = v_host(x, y, 1);
                                double speed = std::sqrt(vx * vx + vy * vy);
                                double intensity = std::min(speed / 0.1, 1.0);

                                uint8_t r =
                                    static_cast<uint8_t>(intensity * 255);
                                uint8_t g = 0;
                                uint8_t b = static_cast<uint8_t>(
                                    (1.0 - intensity) * 255);
                                uint8_t a = 255;

                                pixels[y * Nx + x] =
                                    (r << 24) | (g << 16) | (b << 8) | a;
                            }
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

                    // Sync the "quit" state to all other MPI processes
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