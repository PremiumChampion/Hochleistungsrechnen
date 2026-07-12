#include <Kokkos_Core.hpp>
#include "field.h"

/**
 * Initializes the LBM simulation grid.
 * Sets the macroscopic variables (density `rho`, velocity `v`) and the mesoscopic
 * distribution functions (`f`) to their initial states based on a chosen pattern.
 */
void initialize_lbm(int global_Nx, int global_Ny, int local_Nx, int local_Ny,
                    int offset_x, int offset_y,
                    ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next,
                    InitialisationPattern pattern) {
    
    // Views are allocated by the Simulation constructor.
    // We fill them here. Interior is [1, local_Nx] and [1, local_Ny].
    // The index 0 and local_N+1 are reserved for halo/ghost cells for MPI communication.

    if (pattern == InitialisationPattern::Empty) {
        // Kokkos::parallel_for with MDRangePolicy launches a 2D parallel loop on the GPU/CPU.
        Kokkos::parallel_for("InitFields",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                // D2Q9 Lattice weights (center, 4 orthogonal, 4 diagonal)
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                rho(x, y) = 1.0;     // Resting density
                v(x, y, 0) = 0.0;    // Zero X velocity
                v(x, y, 1) = 0.0;    // Zero Y velocity
                
                // Initialize distribution function f_i to equilibrium at rest (which is just w_i * rho)
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i];
            });
    }
    else if (pattern == InitialisationPattern::Wave) {
        Kokkos::parallel_for("InitFieldsWave",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
                // D2Q9 discrete velocity vectors
                const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
                const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
                
                const double PI = 3.14159265358979323846;
                const double wave_length = global_Nx;
                const double k = 2.0 * PI / wave_length;
                const double amp = 0.05;
                
                // Calculate global coordinates to ensure smooth wave across MPI boundaries
                int gx = offset_x + (x - 1);
                int gy = offset_y + (y - 1);
                
                // Sinusoidal perturbations in density and velocity
                double current_rho = 1.0 + amp * Kokkos::sin(k * gx);
                double vx = amp * Kokkos::sin(k * gx);
                double vy = amp * Kokkos::cos(k * gy) / 2.0;
                
                rho(x, y) = current_rho;
                v(x, y, 0) = vx;
                v(x, y, 1) = vy;
                
                double u_sq = vx * vx + vy * vy;
                // Initialize f to the local equilibrium distribution f^eq(rho, u)
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * vx + cy[i] * vy;
                    // Standard LBM equilibrium formula: w_i * rho * (1 + 3(c.u) + 4.5(c.u)^2 - 1.5(u)^2)
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
                
                // Droplet centers relative to the global simulation domain
                const double center_x = global_Nx / 2.0;
                const double center_y = global_Ny / 2.0;
                const double radius = global_Nx / 10.0;
                const double amp = 0.5;
                
                int gx = offset_x + (x - 1);
                int gy = offset_y + (y - 1);
                double dx = gx - center_x;
                double dy = gy - center_y;
                double dist_sq = dx * dx + dy * dy;
                
                // Gaussian density droplet
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
                
                // Shear wave: X-velocity varies vertically, commonly used for viscosity testing
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

/**
 * Streaming
 * Moves particle distributions to neighboring cells based on their discrete velocities
 * pull scheme (f_out reads from f_in's neighbors) avoids data races in parallel execution
 */
void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
               int offset_x, int offset_y, int global_Nx, int global_Ny,
               CellTypeField cell_type, int local_Nx, int local_Ny) {
    // Discrete velocities as integers for neighbor indexing
    const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
    // Map to the opposite direction (e.g., East (1) opposes West (3)) for bounce-back conditions
    const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    
    Kokkos::parallel_for("StreamingStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            int gx = offset_x + (x - 1);
            int gy = offset_y + (y - 1);
            
            for (int i = 0; i < 9; ++i) {
                // The source cell we are pulling fluid from in direction i
                int src_x = x - cx[i];
                int src_y = y - cy[i];
                
                // Source cell in the global domain (used for global bounds checks)
                int global_src_x = gx - cx[i];
                int global_src_y = gy - cy[i];
                
                bool is_wall = false;
                
                // Check global domain boundaries if global bounce-back is enabled
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
                    // Half-way bounce-back: Particle reflects back where it came from (opp[i])
                    f_out(x, y, i) = f_in(x, y, opp[i]);

                    // Moving Wall / Lid-driven Cavity condition
                    // If the boundary hit is the top global wall, add momentum
                    if (bounce && global_src_y >= global_Ny) {
                        f_out(x, y, i) += 6.0 * w[i] * cx[i] * u_lid;
                    }
                } else {
                    // Normal streaming: Read from local array (which includes halo/ghost cells if src is on boundary)
                    f_out(x, y, i) = f_in(src_x, src_y, i);
                }
            }
        });
}

/**
 * Collision (Relaxation).
 * Implements the BGK (Bhatnagar-Gross-Krook) approximation.
 * Relaxes the particle distributions towards the local macroscopic equilibrium.
 */
void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("CollideStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
            const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
            
            // Read local macroscopic variables (computed in compute_velocity)
            double rho_val = rho(x, y);
            double ux = v(x, y, 0);
            double uy = v(x, y, 1);
            double usq = ux * ux + uy * uy;
            
            for (int i = 0; i < 9; ++i) {
                // Projection of velocity onto lattice direction
                double cu = cx[i] * ux + cy[i] * uy;
                
                // Calculate Equilibrium Distribution (feq) for D2Q9
                double feq = w[i] * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                
                // BGK Collision operator: f_new = f_old - omega * (f_old - feq)
                // Omega is the relaxation parameter inversely proportional to kinematic viscosity.
                f(x, y, i) = f(x, y, i) - omega * (f(x, y, i) - feq);
            }
        });
}

/**
 * Recalculates Macroscopic Properties.
 * computes density and momentum from the distribution functions.
 */
void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("ComputeMacroscopicFields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const double cx[9] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
            const double cy[9] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};
            
            double local_rho = 0.0;
            double local_vx = 0.0;
            double local_vy = 0.0;

            // Sum up distributions to get density and momentum
            for (int i = 0; i < 9; ++i) {
                double f_i = f(x, y, i);
                local_rho += f_i;
                local_vx += f_i * cx[i];
                local_vy += f_i * cy[i];
            }
            
            rho(x, y) = local_rho;
            v(x, y, 0) = local_vx / local_rho;
            v(x, y, 1) = local_vy / local_rho;
        });
}

/**
 * Forces boundary constraints inside the domain (e.g., fluid injection/extraction).
 * Overwrites the distribution functions at source/sink locations.
 */
void apply_sources(DistFuncD2Q9 f, CellTypeField cell_type, int local_Nx, int local_Ny) {
    Kokkos::parallel_for("ApplySources",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            int type = cell_type(x, y);
            const double w[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
            
            if (type == 2) {
                // SOURCE: Force a constant high density (rho = 1.1), velocity = 0
                // This will push fluid outward into the domain.
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 1.1;
            } else if (type == 3) {
                // SINK: Force a constant low density (rho = 0.9), velocity = 0
                // This creates a pressure vacuum that pulls fluid in.
                for (int i = 0; i < 9; ++i)
                    f(x, y, i) = w[i] * 0.9;
            }
        });
}

/**
 * Computes scalar speed (magnitude of velocity) to prepare for visualization.
 */
void velocity_to_speed(int local_Nx, int local_Ny, VectorField2D v, ScalarField2D speed) {
    Kokkos::parallel_for("VelocityToSpeed",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double vx = v(x, y, 0);
            double vy = v(x, y, 1);
            speed(x, y) = Kokkos::sqrt(vx * vx + vy * vy);
        });
}

/**
 * Finds the maximum speed in the local grid using a parallel reduction pattern.
 * Required for normalizing colors dynamically.
 */
double get_max_speed(int local_Nx, int local_Ny, ScalarField2D speed) {
    double max_val = 0.0;
    Kokkos::parallel_reduce("GetMaxSpeed",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &max) {
            if (speed(x, y) > max) max = speed(x, y);
        }, Kokkos::Max<double>(max_val)); // Kokkos reducer ensures thread-safe max computation
    return max_val;
}

/**
 * Renders the speed field as a pixel array mapped to a Blue -> Red color scale.
 */
void speed_to_rgb(int local_Nx, int local_Ny, ScalarField2D speed, PixelField rgb, double max_speed) {
    Kokkos::parallel_for("SpeedToRGB",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double s = 0.0;
            // Normalize speed to [0.0, 1.0] relative to max_speed
            if (max_speed > 1e-8) {
                s = speed(x, y) / max_speed;
            }
            
            if (s > 1.0) {
                // Overexposed (can happen between frame max_speed updates) -> White
                rgb((y - 1) * local_Nx + (x - 1)) = (255 << 24) | (255 << 16) | (255 << 8) | 255;
            } else {
                if (s < 0.0) s = 0.0;
                
                // Blue (s=0) to Red (s=1) colormap interpolation
                uint8_t r = static_cast<uint8_t>(255 * s);
                uint8_t b = static_cast<uint8_t>(255 * (1.0 - s));
                
                // Map to ABGR format used by standard raw rendering
                rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
            }
        });
}

/**
 * Renders the velocity vector field into colors using HSV-to-RGB conversion.
 * Hue corresponds to fluid direction, Value corresponds to speed magnitude.
 */
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
                    // Map the 2D direction angle (-PI to PI) to Hue (0.0 to 1.0)
                    // The a circular hue with west red, north green, east green/blue (turquoise), south blue/pink
                    double theta = Kokkos::atan2(vy, vx);
                    double hue = (theta + 3.14159265358979323846) / (2.0 * 3.14159265358979323846);
                    if (hue < 0.0) hue = 0.0;
                    if (hue >= 1.0) hue = 0.999999;
                    
                    // Standard HSV to RGB conversion logic (Saturation is fixed to 1.0)
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
            
            // Pack RGB data for export/rendering
            rgb((y - 1) * local_Nx + (x - 1)) = (r << 24) | (g << 16) | (b << 8) | 255;
        });
}

/**
 * Renders the density field mapped to a Blue -> Red color scale.
 * Centered around equilibrium density (rho = 1.0).
 */
void density_to_rgb(int local_Nx, int local_Ny, ScalarField2D rho, PixelField rgb_density) {
    Kokkos::parallel_for("DensityToColor",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double r_val = rho(x, y);
            // Remap density from [0.8, 1.2] to intensity [0.0, 1.0]
            double intensity = (r_val - 0.8) / 0.4;
            
            if (intensity > 1.0) {
                // High Density Overexposed -> White
                rgb_density((y - 1) * local_Nx + (x - 1)) = (255 << 24) | (255 << 16) | (255 << 8) | 255;
            } else {
                if (intensity < 0.0) intensity = 0.0;
                
                // Blue (low density) to Red (high density) colormap interpolation
                uint8_t r = static_cast<uint8_t>(intensity * 255.0);
                uint8_t b = static_cast<uint8_t>((1.0 - intensity) * 255.0);
                
                rgb_density((y - 1) * local_Nx + (x - 1)) = (r << 24) | (0 << 16) | (b << 8) | 255;
            }
        });
}