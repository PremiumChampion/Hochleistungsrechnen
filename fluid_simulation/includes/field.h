#include <Kokkos_Core.hpp>
#include <mpi.h>

#ifndef FIELD_H
#define FIELD_H

// Define alias types for cleaner code
using ScalarField2D = Kokkos::View<double **>;
using VectorField2D = Kokkos::View<double **[2]>;
using DistFuncD2Q9 = Kokkos::View<double **[9]>;
using PixelField = Kokkos::View<uint32_t *>;

// Boundary Communication Halo Buffer Types
using HaloBuf = Kokkos::View<double **>;
using HaloBufHost = HaloBuf::host_mirror_type;

enum InitialisationPattern {
    Empty,
    Wave,
    Point,
    Droplet,
    ShearWave,
};

void initialize_lbm(int Nx, int global_Ny, int local_Ny, int offset_y,
                    ScalarField2D &rho, VectorField2D &v, DistFuncD2Q9 &f,
                    DistFuncD2Q9 &f_next, InitialisationPattern pattern);

void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f,
                      int local_Ny);

void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce, double u_lid,
               int offset_y, int global_Ny);

void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega,
             int local_Ny);

void velocity_to_speed(int Nx, int local_Ny, VectorField2D v,
                       ScalarField2D speed);

double get_max_speed(int Nx, int local_Ny, ScalarField2D speed,
                     double prev_max);

void speed_to_rgb(int Nx, int local_Ny, ScalarField2D speed,
                  PixelField rgb_speed, double max_speed);

void velocity_dir_to_rgb(int Nx, int local_Ny, VectorField2D v, PixelField rgb,
                         double max_speed);

void halo_exchange(DistFuncD2Q9 f, int local_Ny, int rank, int size,
                   bool bounce, HaloBuf send_top, HaloBuf recv_top,
                   HaloBuf send_bottom, HaloBuf recv_bottom,
                   HaloBufHost h_send_top, HaloBufHost h_recv_top,
                   HaloBufHost h_send_bottom, HaloBufHost h_recv_bottom);

#endif