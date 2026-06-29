#include "field.h"
#include "simulation.h"
#include <Kokkos_Core.hpp>
#include <SDL3/SDL.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

// Struct for synchronizing interactions across MPI ranks easily
struct SyncState {
    int quit = 0;
    int resized = 0;
    int Nx = 400;
    int Ny = 300;
    double omega = 1.0;
    int paused = 0;
    int cells_changed = 0;
    int reset_fluid = 0;
    int clear_canvas = 0;
    int current_view_mode = 0;
};

// Simple text-based layout loader
bool load_template(const std::string &filename, int &out_Nx, int &out_Ny,
                   std::vector<int> &out_cells) {
    std::ifstream file(filename);
    if (!file.is_open())
        return false;
    std::vector<std::string> lines;
    std::string line;
    int max_width = 0;
    while (std::getline(file, line)) {
        if ((int)line.length() > max_width)
            max_width = line.length();
        lines.push_back(line);
    }
    if (lines.empty())
        return false;
    out_Nx = max_width;
    out_Ny = lines.size();
    out_cells.assign(out_Nx * out_Ny, 0);
    for (int y = 0; y < out_Ny; ++y) {
        for (int x = 0; x < (int)lines[y].length(); ++x) {
            char c = lines[y][x];
            if (c == '#')
                out_cells[y * out_Nx + x] = 1; // Wall
            else if (c == '+')
                out_cells[y * out_Nx + x] = 2; // Source
            else if (c == '-')
                out_cells[y * out_Nx + x] = 3; // Sink
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        SyncState state;
        int scale = 3; // Visual scale multiplier

        // Use unique_ptr to easily swap simulations on the fly when resizing
        auto sim =
            std::make_unique<simulation>(state.Nx, state.Ny, state.omega,
                                         InitialisationPattern::Empty, false);
        std::vector<int> global_cells(state.Nx * state.Ny, 0);

        SDL_Window *window = nullptr;
        SDL_Renderer *renderer = nullptr;
        SDL_Texture *texture = nullptr;

        if (rank == 0) {
            std::cout << "\n===== LBM Interactive Fluid Canvas =====\n";
            std::cout << " [1] DRAW WALL | [2] DRAW SOURCE | [3] DRAW SINK\n";
            std::cout << " [L-Click] Apply Brush | [R-Click] Erase Brush\n";
            std::cout << " [V] Change View (Speed/Direction/Density)\n";
            std::cout << " [S] Cycle Size & Scale Geometry\n";
            std::cout << " [T] Load 'template.txt' \n";
            std::cout << " [C] Clear Geometry | [R] Reset Fluid | [SPACE] "
                         "Pause/Play\n";
            std::cout << " [UP]/[DOWN] Change Omega \n\n";

            SDL_Init(SDL_INIT_VIDEO);
            window = SDL_CreateWindow("Interactive Fluid Simulator",
                                      state.Nx * scale, state.Ny * scale, 0);
            renderer = SDL_CreateRenderer(window, nullptr);
            texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                        SDL_TEXTUREACCESS_STREAMING, state.Nx,
                                        state.Ny);
        }

        int current_draw_mode = 1;
        int current_size_idx = 0;
        std::vector<std::pair<int, int>> sizes = {
            {400, 300}, {200, 150}, {800, 600}};

        while (!state.quit) {
            // Rank 0 handles inputs and populates the synchronization state
            if (rank == 0) {
                state.cells_changed = 0;
                state.reset_fluid = 0;
                state.clear_canvas = 0;
                state.resized = 0;

                SDL_Event e;
                while (SDL_PollEvent(&e)) {
                    if (e.type == SDL_EVENT_QUIT)
                        state.quit = 1;
                    if (e.type == SDL_EVENT_KEY_DOWN) {
                        if (e.key.key == SDLK_1)
                            current_draw_mode = 1;
                        if (e.key.key == SDLK_2)
                            current_draw_mode = 2;
                        if (e.key.key == SDLK_3)
                            current_draw_mode = 3;
                        if (e.key.key == SDLK_V)
                            state.current_view_mode =
                                (state.current_view_mode + 1) % 3;
                        if (e.key.key == SDLK_UP)
                            state.omega = std::min(1.95, state.omega + 0.05);
                        if (e.key.key == SDLK_DOWN)
                            state.omega = std::max(0.5, state.omega - 0.05);
                        if (e.key.key == SDLK_SPACE)
                            state.paused = !state.paused;
                        if (e.key.key == SDLK_C)
                            state.clear_canvas = 1;
                        if (e.key.key == SDLK_R)
                            state.reset_fluid = 1;

                        // Resize AND Scale Geometry using Nearest-Neighbor
                        if (e.key.key == SDLK_S) {
                            int old_Nx = state.Nx;
                            int old_Ny = state.Ny;

                            current_size_idx =
                                (current_size_idx + 1) % sizes.size();
                            state.Nx = sizes[current_size_idx].first;
                            state.Ny = sizes[current_size_idx].second;

                            std::vector<int> scaled_cells(state.Nx * state.Ny,
                                                          0);
                            for (int y = 0; y < state.Ny; ++y) {
                                for (int x = 0; x < state.Nx; ++x) {
                                    int src_x = x * old_Nx / state.Nx;
                                    int src_y = y * old_Ny / state.Ny;
                                    scaled_cells[y * state.Nx + x] =
                                        global_cells[src_y * old_Nx + src_x];
                                }
                            }
                            global_cells = scaled_cells;

                            state.resized = 1;
                            state.cells_changed = 1;
                        }

                        if (e.key.key == SDLK_T) {
                            if (load_template("template.txt", state.Nx,
                                              state.Ny, global_cells)) {
                                std::cout << "Loaded template.txt (" << state.Nx
                                          << "x" << state.Ny << ")\n";
                                int old_Nx = state.Nx;
                                int old_Ny = state.Ny;

                                current_size_idx =
                                    (current_size_idx + 1) % sizes.size();
                                state.Nx = sizes[current_size_idx].first;
                                state.Ny = sizes[current_size_idx].second;

                                std::vector<int> scaled_cells(
                                    state.Nx * state.Ny, 0);
                                for (int y = 0; y < state.Ny; ++y) {
                                    for (int x = 0; x < state.Nx; ++x) {
                                        int src_x = x * old_Nx / state.Nx;
                                        int src_y = y * old_Ny / state.Ny;
                                        scaled_cells[y * state.Nx + x] =
                                            global_cells[src_y * old_Nx +
                                                         src_x];
                                    }
                                }
                                global_cells = scaled_cells;
                                state.resized = 1;
                                state.cells_changed = 1;
                            } else {
                                std::cout << "Could not open 'template.txt' or "
                                             "it is empty.\n";
                            }
                        }
                    }
                }

                // Brush Dragging
                float mx, my;
                uint32_t mouse_state = SDL_GetMouseState(&mx, &my);
                if ((mouse_state & SDL_BUTTON_LMASK) ||
                    (mouse_state & SDL_BUTTON_RMASK)) {
                    int draw_mode = (mouse_state & SDL_BUTTON_RMASK)
                                        ? 0
                                        : current_draw_mode;
                    int cx = static_cast<int>(mx) / scale;
                    int cy = static_cast<int>(my) / scale;
                    int brush_radius =
                        (state.Nx > 500) ? 8 : 4; // Dynamic brush size

                    for (int dy = -brush_radius; dy <= brush_radius; ++dy) {
                        for (int dx = -brush_radius; dx <= brush_radius; ++dx) {
                            if (dx * dx + dy * dy <=
                                brush_radius * brush_radius) {
                                int tx = cx + dx, ty = cy + dy;
                                if (tx >= 0 && tx < state.Nx && ty >= 0 &&
                                    ty < state.Ny) {
                                    global_cells[ty * state.Nx + tx] =
                                        draw_mode;
                                    state.cells_changed = 1;
                                }
                            }
                        }
                    }
                }
            }

            // Sync interactions across MPI ranks via struct
            MPI_Bcast(&state, sizeof(SyncState), MPI_BYTE, 0, MPI_COMM_WORLD);
            if (state.quit)
                break;

            if (state.resized) {
                // Reconstruct the simulation entirely
                sim = std::make_unique<simulation>(
                    state.Nx, state.Ny, state.omega,
                    InitialisationPattern::Empty, false);

                // Scale global array backing on remote MPI processes so they
                // don't segfault on Broadcast
                if (rank != 0) {
                    global_cells.resize(state.Nx * state.Ny);
                }

                if (rank == 0) {
                    SDL_SetWindowSize(window, state.Nx * scale,
                                      state.Ny * scale);
                    SDL_DestroyTexture(texture);
                    texture = SDL_CreateTexture(
                        renderer, SDL_PIXELFORMAT_RGBA8888,
                        SDL_TEXTUREACCESS_STREAMING, state.Nx, state.Ny);
                }
            }

            if (state.clear_canvas) {
                global_cells.assign(state.Nx * state.Ny, 0);
                state.cells_changed = 1;
            }

            sim->omega = state.omega;
            if (state.reset_fluid)
                sim->reset();

            // Distribute Geometry
            if (state.cells_changed) {
                MPI_Bcast(global_cells.data(), state.Nx * state.Ny, MPI_INT, 0,
                          MPI_COMM_WORLD);
                for (int y = 1; y <= sim->local_Ny; ++y) {
                    for (int x = 0; x < state.Nx; ++x) {
                        int global_y = y + sim->offset_y - 1;
                        sim->h_cell_type(x, y) =
                            global_cells[global_y * state.Nx + x];
                    }
                }
                Kokkos::deep_copy(sim->cell_type, sim->h_cell_type);
            }

            // SIMULATION RUN
            if (!state.paused)
                sim->step(5);
            else
                sim->step(0); // Fetch render without updating

            HostPixelField h_pixels;
            if (state.current_view_mode == 0)
                h_pixels = sim->get_global_rgb_speed();
            else if (state.current_view_mode == 1)
                h_pixels = sim->get_global_rgb_direction();
            else
                h_pixels = sim->get_global_rgb_density();

            if (rank == 0) {
                for (int y = 0; y < state.Ny; ++y) {
                    for (int x = 0; x < state.Nx; ++x) {
                        int c_type = global_cells[y * state.Nx + x];
                        if (c_type == 1)
                            h_pixels(y * state.Nx + x) =
                                0x888888FF; // Grey Wall
                        else if (c_type == 2)
                            h_pixels(y * state.Nx + x) =
                                0x00FF00FF; // Green Source
                        else if (c_type == 3)
                            h_pixels(y * state.Nx + x) = 0xFF0000FF; // Red Sink
                    }
                }

                SDL_UpdateTexture(texture, nullptr, h_pixels.data(),
                                  state.Nx * sizeof(uint32_t));
                SDL_RenderClear(renderer);
                SDL_RenderTexture(renderer, texture, nullptr, nullptr);
                SDL_RenderPresent(renderer);

                static int frames = 0;
                if (++frames % 10 == 0) {
                    const char *modes[] = {"ERASE", "WALL", "SOURCE", "SINK"};
                    const char *views[] = {"SPEED", "DIRECTION", "DENSITY"};
                    printf("\rMode: %-6s | View: %-10s | Size: %dx%d | Omega: "
                           "%.2f | Max Speed: %.3f   ",
                           modes[current_draw_mode],
                           views[state.current_view_mode], state.Nx, state.Ny,
                           state.omega, sim->max_speed);
                    fflush(stdout);
                }
            }
        }

        if (rank == 0) {
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            std::cout << "\nExiting Interactive Canvas.\n";
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}