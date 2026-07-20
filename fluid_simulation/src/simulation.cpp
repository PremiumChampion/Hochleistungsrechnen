#include "simulation.h"
#include <cmath>
#include <iostream>

// Include MPI extensions for CUDA-aware MPI queries if using OpenMPI
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

Simulation::Simulation(int _Nx, int _Ny, double _omega,
                       InitialisationPattern pattern, bool _bounce,
                       DecompType decomp)
    : global_Nx(_Nx),
      global_Ny(_Ny),
      omega(_omega),
      bounce(_bounce),
      decomp_type(decomp) {

    // Get the global MPI environment details
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create the logical 1D or 2D grid of processors
    setup_cartesian_topology();

    // Memory Allocation for Simulation Fields
    // +2 to the local dimensions to account for ghost cells on all four edges.
    rho = ScalarField2D("rho", local_Nx + 2, local_Ny + 2);
    speed = ScalarField2D("speed", local_Nx + 2, local_Ny + 2);
    v = VectorField2D("v", local_Nx + 2, local_Ny + 2);
    f = DistFuncD2Q9("f", local_Nx + 2, local_Ny + 2);
    f_next = DistFuncD2Q9("f_next", local_Nx + 2, local_Ny + 2);

    // Cell type determines boundaries (0: Fluid, 1: Wall, 2: Source, 3: Sink)
    cell_type = CellTypeField("cells", local_Nx + 2, local_Ny + 2);
    h_cell_type = Kokkos::create_mirror_view(cell_type);
    Kokkos::deep_copy(cell_type, 0);

    // RGB output buffers (no ghost cells)
    rgb_speed = PixelField("rgb_spd", local_Nx * local_Ny);
    rgb_direction = PixelField("rgb_dir", local_Nx * local_Ny);
    rgb_density = PixelField("rgb_dens", local_Nx * local_Ny);

    // Communication edge halo buffers allocation
    send_n = HaloBuf("send_n", local_Nx, 3);
    recv_n = HaloBuf("recv_n", local_Nx, 3);
    send_s = HaloBuf("send_s", local_Nx, 3);
    recv_s = HaloBuf("recv_s", local_Nx, 3);
    send_e = HaloBuf("send_e", local_Ny, 3);
    recv_e = HaloBuf("recv_e", local_Ny, 3);
    send_w = HaloBuf("send_w", local_Ny, 3);
    recv_w = HaloBuf("recv_w", local_Ny, 3);

    // Create CPU-accessible mirrors for fallback MPI communication
    // Check if CUDA available -> if yes useless, if no communication goes threw
    // these buffers
    h_send_n = Kokkos::create_mirror_view(send_n);
    h_recv_n = Kokkos::create_mirror_view(recv_n);
    h_send_s = Kokkos::create_mirror_view(send_s);
    h_recv_s = Kokkos::create_mirror_view(recv_s);
    h_send_e = Kokkos::create_mirror_view(send_e);
    h_recv_e = Kokkos::create_mirror_view(recv_e);
    h_send_w = Kokkos::create_mirror_view(send_w);
    h_recv_w = Kokkos::create_mirror_view(recv_w);

    // Corner buffers handle exactly 1 cell (1 direction) per corner per rank
    send_corner = Kokkos::View<double *>("send_corner", 4);
    recv_corner = Kokkos::View<double *>("recv_corner", 4);
    h_send_corner = Kokkos::create_mirror_view(send_corner);
    h_recv_corner = Kokkos::create_mirror_view(recv_corner);

    // Check for CUDA-aware MPI support dynamically (only runs once)

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cuda_aware_mpi = MPIX_Query_cuda_support();
    std::cout << "Checking CUDA-aware MPI support..." << endl;
#else
    cuda_aware_mpi = false;
#endif
    if (rank == 0) {
        std::cout << "CUDA-aware MPI: "
                  << (cuda_aware_mpi ? "Enabled (Direct GPU Comm)"
                                     : "Disabled (Using Host Buffers)")
                  << "\n";
    }

    // Initialize the LBM fields to the equilibrium state or a specific pattern
    initialize_lbm(global_Nx, global_Ny, local_Nx, local_Ny, offset_x, offset_y,
                   rho, v, f, f_next, pattern);
}

Simulation::~Simulation() {
    // Clean up the custom MPI communicator
    if (cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm);
    }
}

// MPI Cartesian Topology Setup
void Simulation::setup_cartesian_topology() {
    dims[0] = 0; // x-dimension of processor grid
    dims[1] = 0; // y-dimension of processor grid

    if (decomp_type == DecompType::DECOMP_1D) {
        // Force 1D vertical striping (divide Y axis across all ranks, 1 on X
        // axis)
        dims[0] = 1;
        dims[1] = size;
    } else {
        // Let MPI determine the optimal (most square) 2D layout for the given
        // node count
        MPI_Dims_create(size, 2, dims);
    }

    // Set periodicity based on bounce flag (bounce = non-periodic)
    int periods[2] = {!bounce, !bounce};

    // Create the cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // Retrieve the new rank and coordinates in the grid
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Identify cardinal neighbors (N, S, E, W).
    // Returns MPI_PROC_NULL if hitting a non-periodic boundary.
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_w, &nbr_e);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_s, &nbr_n);

    // Identify diagonal neighbors (NE, NW, SE, SW) using relative coordinates
    nbr_ne = nbr_nw = nbr_se = nbr_sw = MPI_PROC_NULL;
    if (nbr_e != MPI_PROC_NULL && nbr_n != MPI_PROC_NULL) {
        int ne_coords[2] = {coords[0] + 1, coords[1] + 1};
        MPI_Cart_rank(cart_comm, ne_coords, &nbr_ne);
    }
    if (nbr_w != MPI_PROC_NULL && nbr_n != MPI_PROC_NULL) {
        int nw_coords[2] = {coords[0] - 1, coords[1] + 1};
        MPI_Cart_rank(cart_comm, nw_coords, &nbr_nw);
    }
    if (nbr_e != MPI_PROC_NULL && nbr_s != MPI_PROC_NULL) {
        int se_coords[2] = {coords[0] + 1, coords[1] - 1};
        MPI_Cart_rank(cart_comm, se_coords, &nbr_se);
    }
    if (nbr_w != MPI_PROC_NULL && nbr_s != MPI_PROC_NULL) {
        int sw_coords[2] = {coords[0] - 1, coords[1] - 1};
        MPI_Cart_rank(cart_comm, sw_coords, &nbr_sw);
    }

    // Calculate local dimensions and global offsets for this specific rank
    local_Nx = global_Nx / dims[0];
    local_Ny = global_Ny / dims[1];
    offset_x = coords[0] * local_Nx;
    offset_y = coords[1] * local_Ny;

    if (rank == 0) {
        std::cout << "Domain Decomposition: " << dims[0] << " × " << dims[1]
                  << " processor grid\n";
    }
}

// Halo Exchange with ghost cells
void Simulation::halo_exchange() {
    auto l_f = this->f;
    auto l_send_n = this->send_n;
    auto l_send_s = this->send_s;
    auto l_send_e = this->send_e;
    auto l_send_w = this->send_w;
    auto l_send_corner = this->send_corner;

    auto l_recv_n = this->recv_n;
    auto l_recv_s = this->recv_s;
    auto l_recv_e = this->recv_e;
    auto l_recv_w = this->recv_w;
    auto l_recv_corner = this->recv_corner;

    int l_Ny = this->local_Ny;
    int l_Nx = this->local_Nx;

    // Step 1: Pack outgoing distribution functions into send buffers (on GPU to
    // support) D2Q9 directions: 0: rest, 1: E, 2: N, 3: W, 4: S, 5: NE, 6: NW,
    // 7: SW, 8: SE

    // North edge sends Southward moving particles (4, 7, 8)
    // North edge sends Northward moving particles (2, 5, 6)
    Kokkos::parallel_for(
        "pack_n", l_Nx, KOKKOS_LAMBDA(int x) {
            l_send_n(x, 0) = l_f(x + 1, l_Ny, 2);
            l_send_n(x, 1) = l_f(x + 1, l_Ny, 5);
            l_send_n(x, 2) = l_f(x + 1, l_Ny, 6);
        });

    // South edge sends Southward moving particles (4, 7, 8)
    Kokkos::parallel_for(
        "pack_s", l_Nx, KOKKOS_LAMBDA(int x) {
            l_send_s(x, 0) = l_f(x + 1, 1, 4);
            l_send_s(x, 1) = l_f(x + 1, 1, 7);
            l_send_s(x, 2) = l_f(x + 1, 1, 8);
        });

    // East edge sends Eastward moving particles (1, 5, 8)
    Kokkos::parallel_for(
        "pack_e", l_Ny, KOKKOS_LAMBDA(int y) {
            l_send_e(y, 0) = l_f(l_Nx, y + 1, 1);
            l_send_e(y, 1) = l_f(l_Nx, y + 1, 5);
            l_send_e(y, 2) = l_f(l_Nx, y + 1, 8);
        });

    // West edge sends Westward moving particles (3, 6, 7)
    Kokkos::parallel_for(
        "pack_w", l_Ny, KOKKOS_LAMBDA(int y) {
            l_send_w(y, 0) = l_f(1, y + 1, 3);
            l_send_w(y, 1) = l_f(1, y + 1, 6);
            l_send_w(y, 2) = l_f(1, y + 1, 7);
        });
    // Corners send diagonal particles
    Kokkos::parallel_for(
        "pack_corners", 4, KOKKOS_LAMBDA(int c) {
            if (c == 0)
                l_send_corner(c) = l_f(l_Nx, l_Ny, 5); // NE sends NE moving (5)
            if (c == 1)
                l_send_corner(c) = l_f(1, l_Ny, 6); // NW sends NW moving (6)
            if (c == 2)
                l_send_corner(c) = l_f(l_Nx, 1, 8); // SE sends SE moving (8)
            if (c == 3)
                l_send_corner(c) = l_f(1, 1, 7); // SW sends SW moving (7)
        });

    // Wait for packing kernels to finish
    Kokkos::fence();

    // Step 2: Issue Non-Blocking MPI Sends & Receives
    MPI_Request reqs[16];
    int n_req = 0;

    // Helper lambda to queue an Irecv/Isend pair if the neighbor exists
    auto issue_halo = [&](int nbr, int send_tag, int recv_tag, double *send_buf,
                          double *recv_buf, int count) {
        if (nbr != MPI_PROC_NULL) {
            MPI_Irecv(recv_buf, count, MPI_DOUBLE, nbr, recv_tag, cart_comm,
                      &reqs[n_req++]);
            MPI_Isend(send_buf, count, MPI_DOUBLE, nbr, send_tag, cart_comm,
                      &reqs[n_req++]);
        }
    };

    if (cuda_aware_mpi) {
        // Direct GPU-to-GPU Communication (MPI directly reads/writes device
        // pointers)
        issue_halo(nbr_n, 0, 1, send_n.data(), recv_n.data(), local_Nx * 3);
        issue_halo(nbr_s, 1, 0, send_s.data(), recv_s.data(), local_Nx * 3);
        issue_halo(nbr_e, 2, 3, send_e.data(), recv_e.data(), local_Ny * 3);
        issue_halo(nbr_w, 3, 2, send_w.data(), recv_w.data(), local_Ny * 3);

        // use pointer arithmetic (`data() + 0`)
        // otherwise CPU to dereferences index 0 of the gpu memory, which
        // results in a Segfault.
        issue_halo(nbr_ne, 4, 7, send_corner.data() + 0, recv_corner.data() + 0,
                   1);
        issue_halo(nbr_nw, 5, 6, send_corner.data() + 1, recv_corner.data() + 1,
                   1);
        issue_halo(nbr_se, 6, 5, send_corner.data() + 2, recv_corner.data() + 2,
                   1);
        issue_halo(nbr_sw, 7, 4, send_corner.data() + 3, recv_corner.data() + 3,
                   1);

        if (n_req > 0) {
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
        }
    } else {
        // Fallback: Staging via Host Buffers (Legacy / CPU fallback)
        // 2.1 Copy packed GPU buffers to CPU
        Kokkos::deep_copy(h_send_n, send_n);
        Kokkos::deep_copy(h_send_s, send_s);
        Kokkos::deep_copy(h_send_e, send_e);
        Kokkos::deep_copy(h_send_w, send_w);
        Kokkos::deep_copy(h_send_corner, send_corner);

        // 2.2 Transmit via CPU memory
        issue_halo(nbr_n, 0, 1, h_send_n.data(), h_recv_n.data(), local_Nx * 3);
        issue_halo(nbr_s, 1, 0, h_send_s.data(), h_recv_s.data(), local_Nx * 3);
        issue_halo(nbr_e, 2, 3, h_send_e.data(), h_recv_e.data(), local_Ny * 3);
        issue_halo(nbr_w, 3, 2, h_send_w.data(), h_recv_w.data(), local_Ny * 3);

        // Host mirrors can safely be dereferenced by the CPU using
        // '&h_send_corner(0)'
        issue_halo(nbr_ne, 4, 7, &h_send_corner(0), &h_recv_corner(0), 1);
        issue_halo(nbr_nw, 5, 6, &h_send_corner(1), &h_recv_corner(1), 1);
        issue_halo(nbr_se, 6, 5, &h_send_corner(2), &h_recv_corner(2), 1);
        issue_halo(nbr_sw, 7, 4, &h_send_corner(3), &h_recv_corner(3), 1);

        if (n_req > 0) {
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
        }

        // 2.3 Copy received CPU buffers back to GPU
        Kokkos::deep_copy(recv_n, h_recv_n);
        Kokkos::deep_copy(recv_s, h_recv_s);
        Kokkos::deep_copy(recv_e, h_recv_e);
        Kokkos::deep_copy(recv_w, h_recv_w);
        Kokkos::deep_copy(recv_corner, h_recv_corner);
    }

    // Step 3: Unpack received edge data into our ghost cells (on GPU)
    Kokkos::parallel_for(
        "unpack_n", local_Nx, KOKKOS_LAMBDA(int x) {
            l_f(x + 1, l_Ny + 1, 4) = l_recv_n(x, 0);
            l_f(x + 1, l_Ny + 1, 7) = l_recv_n(x, 1);
            l_f(x + 1, l_Ny + 1, 8) = l_recv_n(x, 2);
        });
    Kokkos::parallel_for(
        "unpack_s", local_Nx, KOKKOS_LAMBDA(int x) {
            l_f(x + 1, 0, 2) = l_recv_s(x, 0);
            l_f(x + 1, 0, 5) = l_recv_s(x, 1);
            l_f(x + 1, 0, 6) = l_recv_s(x, 2);
        });
    Kokkos::parallel_for(
        "unpack_e", local_Ny, KOKKOS_LAMBDA(int y) {
            l_f(l_Nx + 1, y + 1, 3) = l_recv_e(y, 0);
            l_f(l_Nx + 1, y + 1, 6) = l_recv_e(y, 1);
            l_f(l_Nx + 1, y + 1, 7) = l_recv_e(y, 2);
        });
    Kokkos::parallel_for(
        "unpack_w", local_Ny, KOKKOS_LAMBDA(int y) {
            l_f(0, y + 1, 1) = l_recv_w(y, 0);
            l_f(0, y + 1, 5) = l_recv_w(y, 1);
            l_f(0, y + 1, 8) = l_recv_w(y, 2);
        });
    Kokkos::parallel_for(
        "unpack_corners", 4, KOKKOS_LAMBDA(int c) {
            if (c == 0)
                l_f(l_Nx + 1, l_Ny + 1, 7) = l_recv_corner(c);
            if (c == 1)
                l_f(0, l_Ny + 1, 8) = l_recv_corner(c);
            if (c == 2)
                l_f(l_Nx + 1, 0, 6) = l_recv_corner(c);
            if (c == 3)
                l_f(0, 0, 5) = l_recv_corner(c);
        });

    // Ensure unpacking is finished before proceeding to physics kernels
    Kokkos::fence();
}

// Core LBM Simulation Loop
void Simulation::step(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        Kokkos::Timer step_timer;

        // 1. Exchange boundary data with neighbors
        halo_exchange();

        double comm_time = step_timer.seconds();
        step_timer.reset();

        // 2. Stream particles to neighboring cells (or bounce back off walls)
        streaming(f, f_next, bounce, u_lid, offset_x, offset_y, global_Nx,
                  global_Ny, cell_type, local_Nx, local_Ny);

        // 3. Swap current and next distribution buffers
        std::swap(f, f_next);

        // 4. Override specific cells with source/sink logic
        apply_sources(f, cell_type, local_Nx, local_Ny);

        // 5. Calculate macroscopic variables (Density and Velocity)
        compute_velocity(rho, v, f, local_Nx, local_Ny);

        // 6. Relax distributions towards equilibrium (BGK Collision)
        collide(f, rho, v, omega, local_Nx, local_Ny);

        Kokkos::fence();
        total_compute_time += step_timer.seconds();
        total_comm_time += comm_time;
    }

    // Prepare visualization buffers once each step ->
    velocity_to_speed(local_Nx, local_Ny, v, speed);

    // Determine the global maximum speed for color mapping normalization
    double current_local_max = get_max_speed(local_Nx, local_Ny, speed);
    double current_global_max = 0.0;
    MPI_Allreduce(&current_local_max, &current_global_max, 1, MPI_DOUBLE,
                  MPI_MAX, cart_comm);

    // Apply an Exponential Moving Average to max_speed.
    // prevents the visualization colors from flickering over time.
    if (max_speed <= 1e-8) {
        max_speed = current_global_max;
    } else {
        double alpha = 1.0 / 30.0; // Approx .5 seconds averaging at 60 FPS
        max_speed = max_speed * (1.0 - alpha) + current_global_max * alpha;
    }

    // Convert physics fields into RGBA uint32_t pixels
    speed_to_rgb(local_Nx, local_Ny, speed, rgb_speed, max_speed);
    velocity_dir_to_rgb(local_Nx, local_Ny, v, rgb_direction, max_speed);
    density_to_rgb(local_Nx, local_Ny, rho, rgb_density);
}

// Data Gathering & Export Utilities
// we collect all simulation data and sehd that
void Simulation::gather_pixels(PixelField local_rgb, HostPixelField &global_rgb,
                               const std::string &name) {
    // Bring local GPU pixel data to the CPU
    auto local_host = Kokkos::create_mirror_view(local_rgb);
    Kokkos::deep_copy(local_host, local_rgb);

    if (rank == 0) {
        // Root rank allocates the full global canvas
        global_rgb = HostPixelField(name, global_Nx * global_Ny);

        // 1. Copy Rank 0's own chunk into the global canvas
        for (int y = 0; y < local_Ny; ++y) {
            for (int x = 0; x < local_Nx; ++x) {
                global_rgb((y + offset_y) * global_Nx + (x + offset_x)) =
                    local_host(y * local_Nx + x);
            }
        }

        // 2. Receive chunks from all other ranks and place them based on their
        // offsets
        for (int r = 1; r < size; ++r) {
            int r_coords[2];
            MPI_Cart_coords(cart_comm, r, 2, r_coords);
            int r_off_x = r_coords[0] * local_Nx;
            int r_off_y = r_coords[1] * local_Ny;

            std::vector<uint32_t> recv_buf(local_Nx * local_Ny);
            MPI_Recv(recv_buf.data(), local_Nx * local_Ny, MPI_UINT32_T, r, 0,
                     cart_comm, MPI_STATUS_IGNORE);

            for (int y = 0; y < local_Ny; ++y) {
                for (int x = 0; x < local_Nx; ++x) {
                    global_rgb((y + r_off_y) * global_Nx + (x + r_off_x)) =
                        recv_buf[y * local_Nx + x];
                }
            }
        }
    } else {
        // Worker ranks just send their chunk to Rank 0
        MPI_Send(local_host.data(), local_Nx * local_Ny, MPI_UINT32_T, 0, 0,
                 cart_comm);
    }
}

HostPixelField Simulation::get_global_rgb_speed() {
    HostPixelField global_rgb;
    gather_pixels(rgb_speed, global_rgb, "global_rgb_speed");
    return global_rgb;
}

HostPixelField Simulation::get_global_rgb_direction() {
    HostPixelField global_rgb;
    gather_pixels(rgb_direction, global_rgb, "global_rgb_dir");
    return global_rgb;
}

HostPixelField Simulation::get_global_rgb_density() {
    HostPixelField global_rgb;
    gather_pixels(rgb_density, global_rgb, "global_rgb_dens");
    return global_rgb;
}

HostVectorField2D Simulation::get_global_velocity() {
    HostVectorField2D global_v;
    auto local_v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(local_v_host, v);

    // Pack 2D velocity vectors into a flat raw double array for MPI
    std::vector<double> send_buf(local_Nx * local_Ny * 2);
    for (int y = 0; y < local_Ny; ++y) {
        for (int x = 0; x < local_Nx; ++x) {
            // Note: read from local interior boundaries [1, local_Nx] to skip
            // ghost cells
            send_buf[(y * local_Nx + x) * 2 + 0] =
                local_v_host(x + 1, y + 1, 0);
            send_buf[(y * local_Nx + x) * 2 + 1] =
                local_v_host(x + 1, y + 1, 1);
        }
    }

    if (rank == 0) {
        global_v = HostVectorField2D("global_v", global_Nx, global_Ny);

        // Map Rank 0's data
        for (int y = 0; y < local_Ny; ++y) {
            for (int x = 0; x < local_Nx; ++x) {
                global_v(x + offset_x, y + offset_y, 0) =
                    send_buf[(y * local_Nx + x) * 2 + 0];
                global_v(x + offset_x, y + offset_y, 1) =
                    send_buf[(y * local_Nx + x) * 2 + 1];
            }
        }

        // Receive and map from other ranks
        std::vector<double> recv_buf(local_Nx * local_Ny * 2);
        for (int r = 1; r < size; ++r) {
            int r_coords[2];
            MPI_Cart_coords(cart_comm, r, 2, r_coords);
            int r_off_x = r_coords[0] * local_Nx;
            int r_off_y = r_coords[1] * local_Ny;

            MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_DOUBLE, r, 0,
                     cart_comm, MPI_STATUS_IGNORE);

            for (int y = 0; y < local_Ny; ++y) {
                for (int x = 0; x < local_Nx; ++x) {
                    global_v(x + r_off_x, y + r_off_y, 0) =
                        recv_buf[(y * local_Nx + x) * 2 + 0];
                    global_v(x + r_off_x, y + r_off_y, 1) =
                        recv_buf[(y * local_Nx + x) * 2 + 1];
                }
            }
        }
    } else {
        MPI_Send(send_buf.data(), send_buf.size(), MPI_DOUBLE, 0, 0, cart_comm);
    }

    return global_v;
}

// Total density over the simulation
double Simulation::get_total_density() {
    double local_mass = 0;
    auto l_rho = rho;

    // Sum density across interior cells only
    Kokkos::parallel_reduce(
        "local_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1},
                                               {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            sum += l_rho(x, y);
        },
        local_mass);

    double global_mass = 0;
    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return global_mass;
}
// Total kinematic energy over the simulation
double Simulation::get_total_kinetic_energy() {
    double local_ke = 0;
    auto l_rho = rho;
    auto l_v = v;

    // E_k = 0.5 * rho * v^2
    Kokkos::parallel_reduce(
        "local_ke",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1},
                                               {local_Nx + 1, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            double vx = l_v(x, y, 0);
            double vy = l_v(x, y, 1);
            sum += 0.5 * l_rho(x, y) * (vx * vx + vy * vy);
        },
        local_ke);

    double global_ke = 0;
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return global_ke;
}

void Simulation::reset() {
    // Re-initialize physics variables to starting states
    initialize_lbm(global_Nx, global_Ny, local_Nx, local_Ny, offset_x, offset_y,
                   rho, v, f, f_next, InitialisationPattern::Empty);
}