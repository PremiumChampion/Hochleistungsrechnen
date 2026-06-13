#include "simulation.h"
#include "field.h"

simulation::simulation(uint _Nx, uint _Ny, double _omega,
                       InitialisationPattern pattern, bool _bounce)
    : Nx(_Nx),
      Ny(_Ny),
      omega(_omega),
      bounce(_bounce) {
    speed = ScalarField2D("particle_speed", Nx, Ny);
    rgb_speed = PixelField("speed_pixels", Nx * Ny);
    rgb_direction = PixelField("direction_pixels", Nx * Ny);
    // rgb = PixelField("speed_pixels", Nx * Ny);
    initialize_lbm(Nx, Ny, rho, v, f, f_next, pattern);
}

void simulation::step(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        streaming(f, f_next, bounce, u_lid);
        std::swap(f, f_next);
        compute_velocity(rho, v, f);
        collide(f, rho, v, omega);
    }
    velocity_to_speed(Nx, Ny, v, speed);
    max_speed = get_max_speed(Nx, Ny, speed, max_speed);
    speed_to_rgb(Nx, Ny, speed, rgb_speed, max_speed);
    velocity_dir_to_rgb(Nx, Ny, v, rgb_direction, max_speed);
}
