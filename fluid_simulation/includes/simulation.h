#ifndef SIMULATION_H
#define SIMULATION_H

#include "field.h"
#include <mpi.h>
#include <vector>
#include <string>

using HostVectorField2D = Kokkos::View<double **[2]>::host_mirror_type;
using HostPixelField    = PixelField::host_mirror_type;
using HostScalarField2D = Kokkos::View<double **>::host_mirror_type;

enum class DecompType { DECOMP_1D, DECOMP_2D };

class Simulation {
public:
    // Global and local dimensions
    int global_Nx, global_Ny;
    int local_Nx, local_Ny;
    int offset_x, offset_y;
    int rank, size;

    // Processor grid
    DecompType decomp_type;
    int dims[2];       // {px, py}
    int coords[2];     // this rank's {x, y} in the grid
    MPI_Comm cart_comm;

    // Neighbor ranks (8 neighbors for D2Q9 diagonal streaming)
    int nbr_n, nbr_s, nbr_e, nbr_w;
    int nbr_ne, nbr_nw, nbr_se, nbr_sw;

    bool bounce;
    double omega;
    double max_speed = 0.0;
    double u_lid = 0.0;

    // Core Fields (sized [local_Nx + 2] x [local_Ny + 2] for ghost cells)
    ScalarField2D rho, speed;
    VectorField2D v;
    DistFuncD2Q9 f, f_next;
    
    // Output Pixels (sized local_Nx * local_Ny)
    PixelField rgb_speed, rgb_direction, rgb_density;

    CellTypeField cell_type;
    CellTypeField::host_mirror_type h_cell_type;

    // Edge halo buffers (3 D2Q9 components per cell)
    HaloBuf send_n, recv_n, send_s, recv_s;
    HaloBuf send_e, recv_e, send_w, recv_w;
    HaloBufHost h_send_n, h_recv_n, h_send_s, h_recv_s;
    HaloBufHost h_send_e, h_recv_e, h_send_w, h_recv_w;

    // Corner halo buffers (1 D2Q9 component per corner cell)
    Kokkos::View<double*> send_corner, recv_corner;  // [4]: NE, NW, SE, SW
    Kokkos::View<double*>::host_mirror_type h_send_corner, h_recv_corner;

    double total_compute_time = 0.0;
    double total_comm_time    = 0.0;

    Simulation(int Nx = 1500, int Ny = 1000, double omega = 1.0,
               InitialisationPattern pattern = InitialisationPattern::Droplet,
               bool bounce = true, DecompType decomp = DecompType::DECOMP_2D);
    ~Simulation();

    void step(size_t num_iterations = 1);
    void reset();
    double get_total_density();
    double get_total_kinetic_energy();

    HostVectorField2D get_global_velocity();
    HostPixelField    get_global_rgb_speed();
    HostPixelField    get_global_rgb_direction();
    HostPixelField    get_global_rgb_density();

    void halo_exchange();
    void setup_cartesian_topology();

private:
    void gather_pixels(PixelField local_rgb, HostPixelField& global_rgb, const std::string& name);
};

// Typedef for backwards compatibility with your main.cpp files
using simulation = Simulation;

#endif // SIMULATION_H