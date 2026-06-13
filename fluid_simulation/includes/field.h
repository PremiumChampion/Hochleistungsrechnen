#include <Kokkos_Core.hpp>

#ifndef FIELD_H
#define FIELD_H
// Define alias types for cleaner code
// We use compile-time dimensions ([2] and [9]) for the last indices.
using ScalarField2D = Kokkos::View<double **>;
using VectorField2D = Kokkos::View<double **[2]>;
using DistFuncD2Q9 = Kokkos::View<double **[9]>;
using PixelField = Kokkos::View<uint32_t *>;

enum InitialisationPattern {
    Empty,
    Wave,
    Point,
    Droplet,
    ShearWave,
};

void initialize_lbm(int Nx, int Ny, ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next,
                    InitialisationPattern pattern);
void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f);
// void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce);
void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce,
               double u_lid = 0.0);
void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega);
void velocity_to_speed(int Nx, int Ny, VectorField2D v, ScalarField2D speed);
double get_max_speed(int Nx, int Ny, ScalarField2D speed, double prev_max);
void speed_to_rgb(int Nx, int Ny, ScalarField2D speed, PixelField rgb_speed,
                  double max_speed);
void velocity_dir_to_rgb(int Nx, int Ny, VectorField2D v, PixelField rgb,
                         double max_speed);
#endif