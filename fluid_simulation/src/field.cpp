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

            // Store the computed density
            rho(x, y) = local_rho;

            // Store the computed velocity components (v = sum / rho)
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
               int offset_y, int global_Ny) {
    const int Nx = f_out.extent(0);
    const int local_Ny = f_out.extent(1) - 2;

    Kokkos::parallel_for(
        "StreamingStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // Standard D2Q9 discrete velocity components (integers for
            // grid indexing)
            const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
            const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            if (!bounce) {
                for (int i = 0; i < 9; ++i) {
                    // Determine the coordinates of the neighboring node
                    // to "pull" from. We subtract cx[i] and cy[i]
                    // because we are looking BACKWARDS to the source
                    // node that sent the particle towards (x, y).

                    // Adding Nx and Ny before the modulo operator
                    // ensures that the dividend is positive, properly
                    // handling wrap-around for periodic boundary
                    // conditions on the left/bottom edges.
                    int src_x = (x - cx[i] + Nx) % Nx;
                    int src_y = y - cy[i];
                    // Copy the distribution function from the source
                    // node
                    f_out(x, y, i) = f_in(src_x, src_y, i);
                }
            } else {
                const double w[9] = {4. / 9.,  1. / 9.,  1. / 9.,
                                     1. / 9.,  1. / 9.,  1. / 36.,
                                     1. / 36., 1. / 36., 1. / 36.};
                const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

                for (int i = 0; i < 9; ++i) {
                    // Determine the coordinates of the neighboring node
                    // to "pull" from.
                    int src_x = x - cx[i];
                    int src_y = y - cy[i];

                    // Check if the source node is outside the domain
                    // (i.e., hitting a wall)
                    int global_src_y = src_y + offset_y - 1;

                    if (src_x < 0 || src_x >= Nx || global_src_y < 0 ||
                        global_src_y >= global_Ny) {
                        // Bounce-back: reflect the distribution
                        // function from the wall
                        f_out(x, y, i) = f_in(x, y, opp[i]);
                        if (global_src_y >= global_Ny) {
                            f_out(x, y, i) += 6.0 * w[i] * cx[i] * u_lid;
                        }
                    } else {
                        // Normal streaming from the source node
                        f_out(x, y, i) = f_in(src_x, src_y, i);
                    }
                }
            }
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