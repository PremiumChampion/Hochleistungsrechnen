#include <Kokkos_Core.hpp>
#include "field.h"

void initialize_lbm(int Nx, int global_Ny, int local_Ny, int offset_y,
                    ScalarField2D &rho, VectorField2D &v, DistFuncD2Q9 &f,
                    DistFuncD2Q9 &f_next, InitialisationPattern pattern) {
    // 1. Density: size (Nx, local_Ny + 2 for ghosts)
    rho = ScalarField2D("density", Nx, local_Ny + 2);
    // 2. Velocity
    v = VectorField2D("velocity", Nx, local_Ny + 2);
    // 3. Distribution function
    f = DistFuncD2Q9("distribution_function", Nx, local_Ny + 2);
    f_next = DistFuncD2Q9("distribution_function_next", Nx, local_Ny + 2);

    if (pattern == InitialisationPattern::Empty)
        Kokkos::parallel_for(
            "InitFields",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, local_Ny + 2}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,
                                     1. / 9.,  1. / 9.,  1. / 36.,
                                     1. / 36., 1. / 36., 1. / 36.};
                rho(x, y) = 1.0;
                v(x, y, 0) = 0.0;
                v(x, y, 1) = 0.0;

                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i];
            });

    if (pattern == InitialisationPattern::Wave)
        Kokkos::parallel_for(
            "InitFieldsWave",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, local_Ny + 2}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,
                                     1. / 9.,  1. / 9.,  1. / 36.,
                                     1. / 36., 1. / 36., 1. / 36.};
                const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                      1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                      1.0, 1.0, -1.0, -1.0};

                // Wave properties
                const double PI = 3.14159265358979323846;
                // One full wave across the X-achsis
                const double wave_length = Nx;
                const double k = 2.0 * PI / wave_length; // Wavenumber

                // Set amplitude for the wave (keep it small, Mach < 0.1 for
                //   LBM
                // stability)
                const double amp = 0.05;

                int global_y = y + offset_y - 1;
                // Define the sinusoidal macroscopic fields
                // The wave travels along the X-axis. Y-velocity remains 0.
                double current_rho = 1.0 + amp * Kokkos::sin(k * x);
                double vx = amp * Kokkos::sin(k * x);
                // double vy = 0;
                double vy = amp * Kokkos::cos(k * global_y) / 2;

                // Store macroscopic variables
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;

                // Initialize the distribution function to the equilibrium
                double u_sq = vx * vx + vy * vy;
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;

                    // D2Q9 Equilibrium Function Formula incorporating the
                    // local perturbed density
                    f(x, y, i) = w[i] * current_rho *
                                 (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });

    if (pattern == InitialisationPattern::Point)
        Kokkos::parallel_for(
            "InitFieldsSimple",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, local_Ny + 2}),
            KOKKOS_LAMBDA(const int x, const int y) {
                rho(x, y) = 0.0;
                v(x, y, 0) = 0.0;
                v(x, y, 1) = 0.0;

                // Set all directions to 0 initially
                for (int i = 0; i < 9; ++i) {
                    f(x, y, i) = 0.0;

                    int global_y = y + offset_y - 1;
                    // Place a single cluster of particles in the middle of the
                    // grid, moving exclusively to the right (direction index 1,
                    // where cx = 1, cy = 0)
                    if (x == Nx / 2 && global_y == global_Ny / 2) {
                        f(x, y, 1) = 1.0;
                    }
                }
            });

    if (pattern == InitialisationPattern::Droplet)
        Kokkos::parallel_for(
            "InitFieldsDroplet",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, local_Ny + 2}),
            KOKKOS_LAMBDA(const int x, const int y) {
                // D2Q9 Parameters
                const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,
                                     1. / 9.,  1. / 9.,  1. / 36.,
                                     1. / 36., 1. / 36., 1. / 36.};
                const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                      1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                      1.0, 1.0, -1.0, -1.0};

                // Droplet properties
                const double center_x = Nx / 2.0;
                const double center_y = global_Ny / 2.0;
                const double radius =
                    Nx / 10.0;          // Controls the size of the droplet
                const double amp = 0.5; // Amplitude of the density bump

                int global_y = y + offset_y - 1;
                double dx = x - center_x;
                double dy = global_y - center_y;
                double dist_sq = dx * dx + dy * dy;

                // Gaussian bump for density: creates a smooth "droplet"
                // of high density
                double current_rho =
                    1.0 + amp * Kokkos::exp(-dist_sq / (2.0 * radius * radius));

                // Start at rest
                double vx = 0.0;
                double vy = 0.0;

                // Store macroscopic variables
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;

                // Initialize the distribution function to the
                // equilibrium
                double u_sq = vx * vx + vy * vy;

                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    f(x, y, i) = w[i] * current_rho *
                                 (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });

    if (pattern == InitialisationPattern::ShearWave)
        Kokkos::parallel_for(
            "InitFieldsShearWave",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, local_Ny + 2}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,
                                     1. / 9.,  1. / 9.,  1. / 36.,
                                     1. / 36., 1. / 36., 1. / 36.};
                const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                      1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                      1.0, 1.0, -1.0, -1.0};

                const double PI = 3.14159265358979323846;
                const double epsilon = 0.05;
                const double current_rho = 1.0;

                // Sinusoidal variation of the velocities ux with the
                // position y
                int global_y = y + offset_y - 1;
                double vx =
                    epsilon * Kokkos::sin(2.0 * PI * global_y / global_Ny);
                double vy = 0.0;

                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;

                double u_sq = vx * vx + vy * vy;

                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    f(x, y, i) = w[i] * current_rho *
                                 (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });
}

void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f,
                      int local_Ny) {
    const int Nx = rho.extent(0);

    Kokkos::parallel_for(
        "ComputeMacroscopicFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // Standard D2Q9 discrete velocity components
            const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                  1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                  1.0, 1.0, -1.0, -1.0};

            double local_rho = 0.0;
            double local_vx = 0.0;
            double local_vy = 0.0;

            // Compute the integrals (as discrete sums)
            for (int i = 0; i < 9; ++i) {
                double f_i = f(x, y, i);
                local_rho += f_i;
                local_vx += f_i * cx[i];
                local_vy += f_i * cy[i];
            }

            // Prevent division by zero and extreme densities causing explosion
            if (local_rho < 0.1)
                local_rho = 0.1;
            if (local_rho > 3.0)
                local_rho = 3.0;

            double speed_sq = local_vx * local_vx + local_vy * local_vy;

            if (speed_sq > 0.16) {
                double scale = Kokkos::sqrt(0.16 / speed_sq);
                local_vx *= scale;
                local_vy *= scale;
            }

            // Store the computed density
            rho(x, y) = local_rho;

            // Store the computed velocity components (v = sum / rho)
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
               int offset_y, int global_Ny, CellTypeField cell_type) {
    const int Nx = f_out.extent(0);
    const int local_Ny = f_out.extent(1) - 2;

    Kokkos::parallel_for(
        "StreamingStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,  1. / 9., 1. / 9.,
                                 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
            const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

            for (int i = 0; i < 9; ++i) {
                int src_x = x - cx[i];
                int src_y = y - cy[i];
                int global_src_y = src_y + offset_y - 1;

                bool is_wall = false;

                // Edge bounce conditions OR inner drawn wall cells (1)
                if (bounce && (src_x < 0 || src_x >= Nx || global_src_y < 0 ||
                               global_src_y >= global_Ny)) {
                    is_wall = true;
                } else {
                    src_x = (src_x + Nx) % Nx; // Periodic wrap on X
                    if (cell_type(src_x, src_y) == 1) {
                        is_wall = true; // User drawn internal wall
                    }
                }

                if (is_wall) {
                    f_out(x, y, i) = f_in(x, y, opp[i]);
                    // Only apply lid velocity if bouncing on the top global
                    // wall
                    if (bounce && global_src_y >= global_Ny) {
                        f_out(x, y, i) += 6.0 * w[i] * cx[i] * u_lid;
                    }
                } else {
                    f_out(x, y, i) = f_in(src_x, src_y, i);
                }
            }
        });
}

void apply_sources(DistFuncD2Q9 f, CellTypeField cell_type, int local_Ny) {
    int Nx = f.extent(0);
    Kokkos::parallel_for(
        "ApplySources",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            int type = cell_type(x, y);
            const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,  1. / 9., 1. / 9.,
                                 1. / 36., 1. / 36., 1. / 36., 1. / 36.};

            // Instead of injecting unbounded mass, we enforce a constant
            // pressure/density boundary.
            if (type == 2) {
                // SOURCE: Constant high density (rho = 1.3), velocity = 0
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 1.3;
            } else if (type == 3) {
                // SINK: Constant low density (rho = 0.7), velocity = 0
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 0.7;
            }
        });
}

void density_to_rgb(int Nx, int local_Ny, ScalarField2D rho,
                    PixelField rgb_density) {
    Kokkos::parallel_for(
        "DensityToColor",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 1}, {static_cast<int>(Nx), static_cast<int>(local_Ny + 1)}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double r_val = rho(x, y);
            // Map density roughly [0.8, 1.2] to [0, 1.0] intensity range
            double intensity = (r_val - 0.8) / 0.4;
            if (intensity < 0.0)
                intensity = 0.0;
            if (intensity > 1.0)
                intensity = 1.0;

            // Blue to Red Colormap
            uint8_t r = static_cast<uint8_t>(intensity * 255.0);
            uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255.0);

            rgb_density((y - 1) * Nx + x) =
                (r << 24) | (0 << 16) | (b << 8) | 255;
        });
}

void halo_exchange(DistFuncD2Q9 f, int local_Ny, int rank, int size,
                   bool bounce, HaloBuf send_top, HaloBuf recv_top,
                   HaloBuf send_bottom, HaloBuf recv_bottom,
                   HaloBufHost h_send_top, HaloBufHost h_recv_top,
                   HaloBufHost h_send_bottom, HaloBufHost h_recv_bottom) {
    int Nx = f.extent(0);

    // Pack directional distributions heading towards neighbors
    Kokkos::parallel_for(
        "pack_top", Nx, KOKKOS_LAMBDA(int x) {
            send_top(x, 0) = f(x, local_Ny, 2);
            send_top(x, 1) = f(x, local_Ny, 5);
            send_top(x, 2) = f(x, local_Ny, 6);
        });
    Kokkos::parallel_for(
        "pack_bottom", Nx, KOKKOS_LAMBDA(int x) {
            send_bottom(x, 0) = f(x, 1, 4);
            send_bottom(x, 1) = f(x, 1, 7);
            send_bottom(x, 2) = f(x, 1, 8);
        });
    Kokkos::fence();

    Kokkos::deep_copy(h_send_top, send_top);
    Kokkos::deep_copy(h_send_bottom, send_bottom);

    int top_neighbor = (rank + 1) % size;
    int bottom_neighbor = (rank - 1 + size) % size;

    // Non-Periodic boundaries nullify corresponding neighbour
    // communications
    if (bounce) {
        if (rank == 0)
            bottom_neighbor = MPI_PROC_NULL;
        if (rank == size - 1)
            top_neighbor = MPI_PROC_NULL;
    }

    MPI_Request reqs[4];
    MPI_Irecv(h_recv_top.data(), Nx * 3, MPI_DOUBLE, top_neighbor, 0,
              MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(h_recv_bottom.data(), Nx * 3, MPI_DOUBLE, bottom_neighbor, 1,
              MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(h_send_top.data(), Nx * 3, MPI_DOUBLE, top_neighbor, 1,
              MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(h_send_bottom.data(), Nx * 3, MPI_DOUBLE, bottom_neighbor, 0,
              MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    Kokkos::deep_copy(recv_top, h_recv_top);
    Kokkos::deep_copy(recv_bottom, h_recv_bottom);

    // Unpack data received directly into ghost cells
    Kokkos::parallel_for(
        "unpack_top", Nx, KOKKOS_LAMBDA(int x) {
            f(x, local_Ny + 1, 4) = recv_top(x, 0);
            f(x, local_Ny + 1, 7) = recv_top(x, 1);
            f(x, local_Ny + 1, 8) = recv_top(x, 2);
        });
    Kokkos::parallel_for(
        "unpack_bottom", Nx, KOKKOS_LAMBDA(int x) {
            f(x, 0, 2) = recv_bottom(x, 0);
            f(x, 0, 5) = recv_bottom(x, 1);
            f(x, 0, 6) = recv_bottom(x, 2);
        });
    Kokkos::fence();
}

void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega,
             int local_Ny) {
    const int Nx = rho.extent(0);

    Kokkos::parallel_for(
        "CollideStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,  1. / 9., 1. / 9.,
                                 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
            const double cx[9] = {0.0, 1.0,  0.0,  -1.0, 0.0,
                                  1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0,  0.0, -1.0,
                                  1.0, 1.0, -1.0, -1.0};

            double local_rho = rho(x, y);
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);

            double u_sq = vx * vx + vy * vy;

            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * vx + cy[i] * vy;
                double f_eq = w[i] * local_rho *
                              (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                f(x, y, i) = f(x, y, i) + omega * (f_eq - f(x, y, i));
            }
        });
}

void velocity_to_speed(int Nx, int local_Ny, VectorField2D v,
                       ScalarField2D speed) {
    Kokkos::parallel_for(
        "compute_the_speed_for_each_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(int x, int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            speed(x, y) = Kokkos::sqrt(vx * vx + vy * vy);
        });
}

double get_max_speed(int Nx, int local_Ny, ScalarField2D speed,
                     double prev_max) {
    Kokkos::parallel_reduce(
        "Find maximum value",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(int x, int y, double &local_max) {
            double s = speed(x, y);
            if (s > local_max)
                local_max = s;
        },
        Kokkos::Max<double>(prev_max));
    return prev_max;
}

void speed_to_rgb(int Nx, int local_Ny, ScalarField2D speed,
                  PixelField rgb_speed, double max_speed) {
    Kokkos::parallel_for(
        "VelocityToColor",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 1}, {static_cast<int>(Nx), static_cast<int>(local_Ny + 1)}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double s = speed(x, y);
            double intensity = (s < max_speed) ? (s / max_speed) : 1.0;

            uint8_t r = static_cast<uint8_t>(intensity * 255.0);
            uint8_t g = 0;
            uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255.0);
            uint8_t a = 255;

            rgb_speed((y - 1) * Nx + x) = (r << 24) | (g << 16) | (b << 8) | a;
        });
}

void velocity_dir_to_rgb(int Nx, int local_Ny, VectorField2D v, PixelField rgb,
                         double max_speed) {
    Kokkos::parallel_for(
        "VelocityDirToColor",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 1}, {static_cast<int>(Nx), static_cast<int>(local_Ny + 1)}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);

            double speed = Kokkos::sqrt(vx * vx + vy * vy);
            double val = (max_speed > 0.0) ? (speed / max_speed) : 0.0;
            if (val > 1.0)
                val = 1.0;

            double angle = Kokkos::atan2(vy, vx);
            const double PI = 3.14159265358979323846;
            double hue_deg = (angle + PI) * (180.0 / PI);
            if (hue_deg >= 360.0)
                hue_deg = 0.0;

            double h_prime = hue_deg / 60.0;

            double temp = h_prime;
            while (temp >= 2.0)
                temp -= 2.0;

            double x_val = val * (1.0 - Kokkos::abs(temp - 1.0));

            double r1 = 0, g1 = 0, b1 = 0;
            if (0 <= h_prime && h_prime < 1) {
                r1 = val;
                g1 = x_val;
                b1 = 0;
            } else if (1 <= h_prime && h_prime < 2) {
                r1 = x_val;
                g1 = val;
                b1 = 0;
            } else if (2 <= h_prime && h_prime < 3) {
                r1 = 0;
                g1 = val;
                b1 = x_val;
            } else if (3 <= h_prime && h_prime < 4) {
                r1 = 0;
                g1 = x_val;
                b1 = val;
            } else if (4 <= h_prime && h_prime < 5) {
                r1 = x_val;
                g1 = 0;
                b1 = val;
            } else if (5 <= h_prime && h_prime <= 6) {
                r1 = val;
                g1 = 0;
                b1 = x_val;
            }

            uint8_t r = static_cast<uint8_t>(r1 * 255.0);
            uint8_t g = static_cast<uint8_t>(g1 * 255.0);
            uint8_t b = static_cast<uint8_t>(b1 * 255.0);
            uint8_t a = 255;

            rgb((y - 1) * Nx + x) = (r << 24) | (g << 16) | (b << 8) | a;
        });
}


void initialize_lbm_2d(int global_Nx, int global_Ny, int local_Nx, int local_Ny,
                       int offset_x, int offset_y,
                       ScalarField2D &rho, VectorField2D &v,
                       DistFuncD2Q9 &f, DistFuncD2Q9 &f_next,
                       InitialisationPattern pattern) {
    
    // Note: Views are already allocated by the Simulation2D constructor.
    // We just fill them here. Interior is [1, local_Nx] and [1, local_Ny].

    if (pattern == InitialisationPattern::Empty) {
        Kokkos::parallel_for(
            "InitFields2D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                rho(x, y) = 1.0;
                v(x, y, 0) = 0.0;
                v(x, y, 1) = 0.0;
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i];
            });
    }
    else if (pattern == InitialisationPattern::Wave) {
        Kokkos::parallel_for(
            "InitFieldsWave2D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
                
                const double PI = 3.14159265358979323846;
                const double wave_length = global_Nx;
                const double k = 2.0 * PI / wave_length;
                const double amp = 0.05;
                
                int gx = offset_x + (x - 1);
                int gy = offset_y + (y - 1);
                
                double current_rho = 1.0 + amp * Kokkos::sin(k * gx);
                double vx = amp * Kokkos::sin(k * gx);
                double vy = amp * Kokkos::cos(k * gy) / 2.0;
                
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;
                
                double u_sq = vx * vx + vy * vy;
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    f(x, y, i) = w[i] * current_rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });
    }
    else if (pattern == InitialisationPattern::Droplet) {
        Kokkos::parallel_for(
            "InitFieldsDroplet2D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
                
                const double center_x = global_Nx / 2.0;
                const double center_y = global_Ny / 2.0;
                const double radius = global_Nx / 10.0;
                const double amp = 0.5;
                
                int gx = offset_x + (x - 1);
                int gy = offset_y + (y - 1);
                double dx = gx - center_x;
                double dy = gy - center_y;
                double dist_sq = dx * dx + dy * dy;
                
                double current_rho = 1.0 + amp * Kokkos::exp(-dist_sq / (2.0 * radius * radius));
                double vx = 0.0;
                double vy = 0.0;
                
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;
                
                double u_sq = vx * vx + vy * vy;
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    f(x, y, i) = w[i] * current_rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });
    }
    else if (pattern == InitialisationPattern::ShearWave) {
        Kokkos::parallel_for(
            "InitFieldsShearWave2D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
                const double PI = 3.14159265358979323846;
                const double epsilon = 0.05;
                const double current_rho = 1.0;
                
                int gx = offset_x + (x - 1);
                int gy = offset_y + (y - 1);
                
                double vx = epsilon * Kokkos::sin(2.0 * PI * gy / global_Ny);
                double vy = 0.0;
                
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;
                
                double u_sq = vx * vx + vy * vy;
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    f(x, y, i) = w[i] * current_rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                }
            });
    }
}


void streaming_2d(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
                  int offset_x, int offset_y, int global_Nx, int global_Ny,
                  CellTypeField cell_type, int local_Nx, int local_Ny) {
    const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
    const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    
    Kokkos::parallel_for(
        "StreamingStep2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            int gx = offset_x + (x - 1);
            int gy = offset_y + (y - 1);
            
            for (int i = 0; i < 9; ++i) {
                int src_x = x - cx[i];
                int src_y = y - cy[i];
                
                int global_src_x = gx - cx[i];
                int global_src_y = gy - cy[i];
                
                bool is_wall = false;
                
                // Check global domain boundaries if bounce is enabled
                if (bounce && (global_src_x < 0 || global_src_x >= global_Nx || 
                               global_src_y < 0 || global_src_y >= global_Ny)) {
                    is_wall = true;
                } else {
                    // Check for user-drawn internal walls. 
                    // Only check if src is within local interior bounds (avoid reading un-exchanged ghost cell_type)
                    if (src_x >= 1 && src_x <= local_Nx && 
                        src_y >= 1 && src_y <= local_Ny && 
                        cell_type(src_x, src_y) == 1) {
                        is_wall = true;
                    }
                }
                
                if (is_wall) {
                    f_out(x, y, i) = f_in(x, y, opp[i]);
                    // Apply lid velocity at the top global wall
                    if (bounce && global_src_y >= global_Ny) {
                        f_out(x, y, i) += 6.0 * w[i] * cx[i] * u_lid;
                    }
                } else {
                    // Read from local array (which includes halo/ghost cells if src is on boundary)
                    f_out(x, y, i) = f_in(src_x, src_y, i);
                }
            }
        });
}


void collide_2d(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega, int local_Nx, int local_Ny) {
    Kokkos::parallel_for(
        "CollideStep2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
            const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
            
            double rho_val = rho(x, y);
            double ux = v(x, y, 0);
            double uy = v(x, y, 1);
            double usq = ux * ux + uy * uy;
            
            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * ux + cy[i] * uy;
                double feq = w[i] * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                f(x, y, i) = f(x, y, i) - omega * (f(x, y, i) - feq);
            }
        });
}


void compute_velocity_2d(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f, int local_Nx, int local_Ny) {
    Kokkos::parallel_for(
        "ComputeMacroscopicFields2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
            
            double local_rho = 0.0;
            double local_vx = 0.0;
            double local_vy = 0.0;
            
            for (int i = 0; i < 9; ++i) {
                double f_i = f(x, y, i);
                local_rho += f_i;
                local_vx += f_i * cx[i];
                local_vy += f_i * cy[i];
            }
            
            if (local_rho < 0.1) local_rho = 0.1;
            if (local_rho > 3.0) local_rho = 3.0;
            
            double speed_sq = local_vx * local_vx + local_vy * local_vy;
            if (speed_sq > 0.16) {
                double scale = Kokkos::sqrt(0.16 / speed_sq);
                local_vx *= scale;
                local_vy *= scale;
            }
            
            rho(x, y) = local_rho;
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

void apply_sources_2d(DistFuncD2Q9 f, CellTypeField cell_type, int local_Nx, int local_Ny) {
    Kokkos::parallel_for(
        "ApplySources2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            int type = cell_type(x, y);
            const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
            
            if (type == 2) {
                // SOURCE: Constant high density (rho = 1.3), velocity = 0
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 1.3;
            } else if (type == 3) {
                // SINK: Constant low density (rho = 0.7), velocity = 0
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 0.7;
            }
        });
}


void velocity_to_speed_2d(int local_Nx, int local_Ny, VectorField2D v, ScalarField2D speed) {
    Kokkos::parallel_for(
        "VelocityToSpeed2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            speed(x, y) = Kokkos::sqrt(vx * vx + vy * vy);
        });
}

double get_max_speed_2d(int local_Nx, int local_Ny, ScalarField2D speed, double prev_max) {
    double max_val = prev_max;
    Kokkos::parallel_reduce(
        "GetMaxSpeed2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &max) {
            if (speed(x, y) > max) max = speed(x, y);
        },
        Kokkos::Max<double>(max_val));
    return max_val;
}

void speed_to_rgb_2d(int local_Nx, int local_Ny, ScalarField2D speed, PixelField rgb, double max_speed) {
    Kokkos::parallel_for(
        "SpeedToRGB2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double s = speed(x, y) / max_speed;
            if (s < 0.0) s = 0.0;
            if (s > 1.0) s = 1.0;
            
            // Blue to Red colormap
            uint8_t r = static_cast<uint8_t>(255 * s);
            uint8_t b = static_cast<uint8_t>(255 * (1.0 - s));
            
            rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
        });
}

void velocity_dir_to_rgb_2d(int local_Nx, int local_Ny, VectorField2D v, PixelField rgb, double max_speed) {
    Kokkos::parallel_for(
        "VelocityDirToRGB2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            double mag = Kokkos::sqrt(vx * vx + vy * vy);
            
            uint8_t r = 0, g = 0, b = 0;
            if (mag > 1e-6 && max_speed > 1e-6) {
                double norm_vx = vx / max_speed;
                double norm_vy = vy / max_speed;
                double norm_mag = mag / max_speed;
                
                r = static_cast<uint8_t>(255 * Kokkos::min(1.0, Kokkos::max(0.0, norm_vx * 0.5 + 0.5)));
                g = static_cast<uint8_t>(255 * Kokkos::min(1.0, Kokkos::max(0.0, norm_vy * 0.5 + 0.5)));
                b = static_cast<uint8_t>(255 * Kokkos::min(1.0, norm_mag));
            }
            
            rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (g << 16) | (b << 8) | 255;
        });
}

void density_to_rgb_2d(int local_Nx, int local_Ny, ScalarField2D rho, PixelField rgb_density) {
    Kokkos::parallel_for(
        "DensityToColor2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double r_val = rho(x, y);
            // Map density roughly [0.8, 1.2] to [0, 1.0] intensity range
            double intensity = (r_val - 0.8) / 0.4;
            if (intensity < 0.0) intensity = 0.0;
            if (intensity > 1.0) intensity = 1.0;
            
            // Blue to Red Colormap
            uint8_t r = static_cast<uint8_t>(intensity * 255.0);
            uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255.0);
            
            rgb_density((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
        });
}