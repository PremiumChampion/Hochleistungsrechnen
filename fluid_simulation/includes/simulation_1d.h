#ifndef SIMULATION_1D_H
#define SIMULATION_1D_H

#include "field.h"
#include <mpi.h>
#include <vector>

using HostVectorField2D = Kokkos::View<double **[2]>::host_mirror_type;
using HostPixelField     = PixelField::host_mirror_type;
using HostScalarField2D  = Kokkos::View<double **>::host_mirror_type;

class Simulation1D {
public:
    int global_Nx, global_Ny, local_Ny, offset_y, rank, size;
    bool bounce = true;
    double omega = 1.0;
    double max_speed = 0;
    double u_lid = 0.0;

    ScalarField2D rho, speed;
    VectorField2D v;
    DistFuncD2Q9 f, f_next;
    PixelField rgb_speed, rgb_direction, rgb_density;

    CellTypeField cell_type;
    CellTypeField::host_mirror_type h_cell_type;

    // Halo buffers (top/bottom only — 1D decomposition)
    HaloBuf send_top, recv_top, send_bottom, recv_bottom;
    HaloBufHost h_send_top, h_recv_top, h_send_bottom, h_recv_bottom;

    double total_compute_time = 0.0;
    double total_comm_time    = 0.0;

    Simulation1D(int Nx = 1500, int Ny = 1000, double omega = 1.0,
                 InitialisationPattern pattern = InitialisationPattern::Droplet,
                 bool bounce = true);
    ~Simulation1D() = default;

    void   step(size_t num_iterations = 1);
    void   reset();
    double get_total_density();
    double get_total_kinetic_energy();

    HostVectorField2D get_global_velocity();
    HostPixelField    get_global_rgb_speed();
    HostPixelField    get_global_rgb_direction();
    HostPixelField    get_global_rgb_density();
};

#endif // SIMULATION_1D_H