#include <Kokkos_Core.hpp>
#include "field.h"

void initialize_lbm(int global_Nx, int global_Ny, int local_Nx, int local_Ny,
                    int offset_x, int offset_y,
                    ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next,
                    InitialisationPattern pattern) {
    
    // Views are allocated by the Simulation constructor.
    // We fill them here. Interior is [1, local_Nx] and [1, local_Ny].

    if (pattern == InitialisationPattern::Empty) {
        Kokkos::parallel_for("InitFields",
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
        Kokkos::parallel_for("InitFieldsWave",
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
        Kokkos::parallel_for("InitFieldsDroplet",
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
        Kokkos::parallel_for("InitFieldsShearWave",
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


void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
               int offset_x, int offset_y, int global_Nx, int global_Ny,
               CellTypeField cell_type, int local_Nx, int local_Ny) {
    const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
    const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    
    Kokkos::parallel_for("StreamingStep",
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
                    // Only check if src is within local interior bounds
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


void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("CollideStep",
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


void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("ComputeMacroscopicFields",
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
            
            rho(x, y) = local_rho;
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

void apply_sources(DistFuncD2Q9 f, CellTypeField cell_type, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("ApplySources",
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

void velocity_to_speed(int local_Nx, int local_Ny, VectorField2D v, ScalarField2D speed) {
    Kokkos::parallel_for("VelocityToSpeed",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            speed(x, y) = Kokkos::sqrt(vx * vx + vy * vy);
        });
}

double get_max_speed(int local_Nx, int local_Ny, ScalarField2D speed) {
    double max_val = 0.0;
    Kokkos::parallel_reduce("GetMaxSpeed",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &max) {
            if (speed(x, y) > max) max = speed(x, y);
        }, Kokkos::Max<double>(max_val));
    return max_val;
}

void speed_to_rgb(int local_Nx, int local_Ny, ScalarField2D speed, PixelField rgb, double max_speed) {
    Kokkos::parallel_for("SpeedToRGB",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double s = 0.0;
            if (max_speed > 1e-8) {
                s = speed(x, y) / max_speed;
            }
            
            if (s > 1.0) {
                // Overexposed -> White
                rgb((y - 1) * local_Nx + (x - 1)) = (255 << 24) | (255 << 16) | (255 << 8) | 255;
            } else {
                if (s < 0.0) s = 0.0;
                
                // Blue to Red colormap
                uint8_t r = static_cast<uint8_t>(255 * s);
                uint8_t b = static_cast<uint8_t>(255 * (1.0 - s));
                
                rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
            }
        });
}

void velocity_dir_to_rgb(int local_Nx, int local_Ny, VectorField2D v, PixelField rgb, double max_speed) {
    Kokkos::parallel_for("VelocityDirToRGB",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            double mag = Kokkos::sqrt(vx * vx + vy * vy);
            
            uint8_t r = 0, g = 0, b = 0;
            if (mag > 1e-6 && max_speed > 1e-6) {
                double v_val = mag / max_speed;
                
                if (v_val > 1.0) {
                    // Overexposed -> White
                    r = 255; g = 255; b = 255;
                } else {
                    // Map the 2D direction angle to Hue
                    double theta = Kokkos::atan2(vy, vx);
                    double hue = (theta + 3.14159265358979323846) / (2.0 * 3.14159265358979323846);
                    if (hue < 0.0) hue = 0.0;
                    if (hue >= 1.0) hue = 0.999999;
                    
                    double h_val = hue * 6.0;
                    int i = static_cast<int>(h_val);
                    double f = h_val - i;
                    
                    double p = 0.0; // Saturation is always 1.0, so p = 0
                    double q = v_val * (1.0 - f);
                    double t = v_val * f;
                    
                    double r_val = 0.0, g_val = 0.0, b_val = 0.0;
                    if (i == 0)      { r_val = v_val; g_val = t;     b_val = p; }
                    else if (i == 1) { r_val = q;     g_val = v_val; b_val = p; }
                    else if (i == 2) { r_val = p;     g_val = v_val; b_val = t; }
                    else if (i == 3) { r_val = p;     g_val = q;     b_val = v_val; }
                    else if (i == 4) { r_val = t;     g_val = p;     b_val = v_val; }
                    else             { r_val = v_val; g_val = p;     b_val = q; }
                    
                    r = static_cast<uint8_t>(255 * r_val);
                    g = static_cast<uint8_t>(255 * g_val);
                    b = static_cast<uint8_t>(255 * b_val);
                }
            }
            
            rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (g << 16) | (b << 8) | 255;
        });
}

void density_to_rgb(int local_Nx, int local_Ny, ScalarField2D rho, PixelField rgb_density) {
    Kokkos::parallel_for("DensityToColor",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double r_val = rho(x, y);
            double intensity = (r_val - 0.8) / 0.4;
            
            if (intensity > 1.0) {
                // Overexposed -> White
                rgb_density((y - 1) * local_Nx + (x - 1)) = (255 << 24) | (255 << 16) | (255 << 8) | 255;
            } else {
                if (intensity < 0.0) intensity = 0.0;
                
                uint8_t r = static_cast<uint8_t>(intensity * 255.0);
                uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255.0);
                
                rgb_density((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
            }
        });
}