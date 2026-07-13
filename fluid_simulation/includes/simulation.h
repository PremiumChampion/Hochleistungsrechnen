#ifndef SIMULATION_H
#define SIMULATION_H

#include "field.h"
#include <mpi.h>
#include <string>
#include <vector>

using HostVectorField2D = Kokkos::View<double **[2]>::host_mirror_type;
using HostPixelField = PixelField::host_mirror_type;
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
    // 1d or 2d decomposition
    DecompType decomp_type;
    int dims[2];        // {px, py}
    int coords[2];      // this rank's {x, y} in the grid
    MPI_Comm cart_comm; // the communication "interface/backbone"
    bool cuda_aware_mpi = false;

    // Neighbor ranks (8 neighbors for D2Q9 diagonal streaming)
    int nbr_n, nbr_s, nbr_e, nbr_w;
    int nbr_ne, nbr_nw, nbr_se, nbr_sw;

    // enforce bouncing off walls
    bool bounce;
    // Relaxation Parameter (inverse of the relaxation time τ). It controls how
    // fast the current distribution f_i relaxes toward the equilibrium f_i_eq.
    double omega;
    // if bounce from top wall, injext momentum: + 6.0 * w_i * cx_i * u_lid
    // set to 0 to disable
    double u_lid = 0.0;

    // Core Fields (sized [local_Nx + 2] x [local_Ny + 2] for ghost cells)
    ScalarField2D rho; // Macroscopic Density.
    VectorField2D v;   // Macroscopic Velocity vector.
    //  Particle Distribution Function array of 9 numbers per grid
    //  cell. It represents the probability or density of fluid
    //  particles moving in direction i at that specific cell.
    DistFuncD2Q9 f, f_next;

    // Visualisation parameter
    ScalarField2D speed; // scalar magnitude of the fluid's velocity
    // allows to perform relative mapping, slow fluids dark, fast fluids bright)
    double max_speed = 0.0;
    // Output Pixels (sized local_Nx * local_Ny)
    PixelField rgb_speed, rgb_direction, rgb_density;

    // stores if a cell is a sink, drain, wall
    // used for customising the simulation
    CellTypeField cell_type;
    CellTypeField::host_mirror_type h_cell_type;

    // Edge halo buffers (3 D2Q9 components per cell)
    HaloBuf send_n, recv_n, send_s, recv_s;
    HaloBuf send_e, recv_e, send_w, recv_w;
    HaloBufHost h_send_n, h_recv_n, h_send_s, h_recv_s;
    HaloBufHost h_send_e, h_recv_e, h_send_w, h_recv_w;

    // Corner halo buffers (1 D2Q9 component per corner cell)
    Kokkos::View<double *> send_corner, recv_corner; // [4]: NE, NW, SE, SW
    Kokkos::View<double *>::host_mirror_type h_send_corner, h_recv_corner;

    // benchmarks
    double total_compute_time = 0.0;
    double total_comm_time = 0.0;

    Simulation(int Nx = 1500, int Ny = 1000, double omega = 1.0,
               InitialisationPattern pattern = InitialisationPattern::Droplet,
               bool bounce = true, DecompType decomp = DecompType::DECOMP_2D);
    ~Simulation();

    // performs x steps of the simulation, then creates a visualisation and
    // gathers all ixels at rank 0
    void step(size_t num_iterations = 1);
    // reset the simulation
    void reset();
    // computes the total dennsity in the simulation, should be constant if no
    // fluid enters/leaves the simulation
    double get_total_density();
    // computes the total kinematic energy in the simulation
    double get_total_kinetic_energy();

    // members for accessing the collected global data
    HostVectorField2D get_global_velocity();
    // visualises just the speeed in a blue/red fashion
    HostPixelField get_global_rgb_speed();
    // visualises the speed and direction in a color circle
    HostPixelField get_global_rgb_direction();
    // visualises the density (rho)
    HostPixelField get_global_rgb_density();

    // performs the ghost cell exchange between neighbouring nodes
    void halo_exchange();
    // setup for the halo exchange, creates cart_comm and the correct topology
    // of the simulation.
    void setup_cartesian_topology();

  private:
    // collects pixel from the other nodes.
    void gather_pixels(PixelField local_rgb, HostPixelField &global_rgb,
                       const std::string &name);
};

// Typedef for backwards compatibility
using simulation = Simulation;

#endif // SIMULATION_H