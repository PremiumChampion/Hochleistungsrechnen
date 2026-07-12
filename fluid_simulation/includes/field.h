#include <Kokkos_Core.hpp>
#include <mpi.h>

#ifndef FIELD_H
#define FIELD_H

// Define alias types for cleaner code
using ScalarField2D = Kokkos::View<double **>;
using VectorField2D = Kokkos::View<double **[2]>;
using DistFuncD2Q9 = Kokkos::View<double **[9]>;
using PixelField = Kokkos::View<uint32_t *>;
using CellTypeField = Kokkos::View<int **>; // 0: Fluid, 1: Wall, 2: Source, 3: Sink

// Boundary Communication Halo Buffer Types
using HaloBuf = Kokkos::View<double **>;
using HaloBufHost = HaloBuf::host_mirror_type;

enum InitialisationPattern { Empty, Wave, Point, Droplet, ShearWave };

void initialize_lbm(int global_Nx, int global_Ny, int local_Nx, int local_Ny,
                    int offset_x, int offset_y,
                    ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next,
                    InitialisationPattern pattern);

void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f,
                      int local_Nx, int local_Ny);

void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out, bool bounce,
               double u_lid, int offset_x, int offset_y,
               int global_Nx, int global_Ny, CellTypeField cell_type,
               int local_Nx, int local_Ny);

void collide(DistFuncD2Q9 f, ScalarField2D rho, VectorField2D v, double omega,
             int local_Nx, int local_Ny);

void apply_sources(DistFuncD2Q9 f, CellTypeField cell_type, int local_Nx, int local_Ny);

void velocity_to_speed(int local_Nx, int local_Ny, VectorField2D v,
                       ScalarField2D speed);

double get_max_speed(int local_Nx, int local_Ny, ScalarField2D speed);

void speed_to_rgb(int local_Nx, int local_Ny, ScalarField2D speed,
                  PixelField rgb_speed, double max_speed);

void velocity_dir_to_rgb(int local_Nx, int local_Ny, VectorField2D v, PixelField rgb,
                         double max_speed);

void density_to_rgb(int local_Nx, int local_Ny, ScalarField2D rho,
                    PixelField rgb_density);

#endif