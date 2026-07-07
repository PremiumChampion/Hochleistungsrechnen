#include "simulation_2d.h"
#include <cmath>
#include <iostream>

Simulation2D::Simulation2D(int _Nx, int _Ny, double _omega,
                           InitialisationPattern pattern, bool _bounce)
    : global_Nx(_Nx),
      global_Ny(_Ny),
      omega(_omega),
      bounce(_bounce) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setup_cartesian_topology();

    // Local domain sizes (assumes even division; extend for remainders if
    // needed)
    local_Nx = global_Nx / dims[0];
    local_Ny = global_Ny / dims[1];
    offset_x = coords[0] * local_Nx;
    offset_y = coords[1] * local_Ny;

    // Fields with ghost cells on ALL FOUR sides: (local_Nx+2) × (local_Ny+2)
    // Interior indices: x ∈ [1, local_Nx], y ∈ [1, local_Ny]
    rho = ScalarField2D("rho", local_Nx + 2, local_Ny + 2);
    speed = ScalarField2D("speed", local_Nx + 2, local_Ny + 2);
    v = VectorField2D("v", local_Nx + 2, local_Ny + 2);
    f = DistFuncD2Q9("f", local_Nx + 2, local_Ny + 2);
    f_next = DistFuncD2Q9("f_next", local_Nx + 2, local_Ny + 2);
    cell_type = CellTypeField("cells", local_Nx + 2, local_Ny + 2);
    h_cell_type = Kokkos::create_mirror_view(cell_type);
    Kokkos::deep_copy(cell_type, 0);

    // RGB output (no ghost cells)
    rgb_speed = PixelField("rgb_spd", local_Nx * local_Ny);
    rgb_direction = PixelField("rgb_dir", local_Nx * local_Ny);
    rgb_density = PixelField("rgb_dens", local_Nx * local_Ny);

    // Edge halo buffers
    // N/S: local_Nx cells × 3 components
    send_n = HaloBuf("send_n", local_Nx, 3);
    recv_n = HaloBuf("recv_n", local_Nx, 3);
    send_s = HaloBuf("send_s", local_Nx, 3);
    recv_s = HaloBuf("recv_s", local_Nx, 3);
    // E/W: local_Ny cells × 3 components
    send_e = HaloBuf("send_e", local_Ny, 3);
    recv_e = HaloBuf("recv_e", local_Ny, 3);
    send_w = HaloBuf("send_w", local_Ny, 3);
    recv_w = HaloBuf("recv_w", local_Ny, 3);

    h_send_n = Kokkos::create_mirror_view(send_n);
    h_recv_n = Kokkos::create_mirror_view(recv_n);
    h_send_s = Kokkos::create_mirror_view(send_s);
    h_recv_s = Kokkos::create_mirror_view(recv_s);
    h_send_e = Kokkos::create_mirror_view(send_e);
    h_recv_e = Kokkos::create_mirror_view(recv_e);
    h_send_w = Kokkos::create_mirror_view(send_w);
    h_recv_w = Kokkos::create_mirror_view(recv_w);

    // Corner buffers: 4 sends + 4 receives (one double each)
    send_corner = Kokkos::View<double *>("send_corner", 4);
    recv_corner = Kokkos::View<double *>("recv_corner", 4);
    h_send_corner = Kokkos::create_mirror_view(send_corner);
    h_recv_corner = Kokkos::create_mirror_view(recv_corner);

    // Initialize LBM (needs modified initializer that uses local_Nx, offset_x)
    initialize_lbm_2d(global_Nx, global_Ny, local_Nx, local_Ny, offset_x,
                      offset_y, rho, v, f, f_next, pattern);
}

// ============================================================================
// Cartesian Topology Setup
// ============================================================================
void Simulation2D::setup_cartesian_topology() {
    dims[0] = 0;
    dims[1] = 0;
    // Let MPI choose a near-square processor grid
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0}; // non-periodic by default
    if (!bounce) {
        periods[0] = 1; // periodic x if no bounce
        periods[1] = 1; // periodic y if no bounce
    }

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Shift along dimension 0 (x) → east/west neighbors
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_w, &nbr_e);
    // Shift along dimension 1 (y) → north/south neighbors
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_s, &nbr_n);

    // Compute diagonal neighbors manually
    // NE: east neighbor's north neighbor (or north neighbor's east neighbor)
    if (nbr_e != MPI_PROC_NULL && nbr_n != MPI_PROC_NULL) {
        int ne_coords[2] = {coords[0] + 1, coords[1] + 1};
        MPI_Cart_rank(cart_comm, ne_coords, &nbr_ne);
    } else {
        nbr_ne = MPI_PROC_NULL;
    }
    if (nbr_w != MPI_PROC_NULL && nbr_n != MPI_PROC_NULL) {
        int nw_coords[2] = {coords[0] - 1, coords[1] + 1};
        MPI_Cart_rank(cart_comm, nw_coords, &nbr_nw);
    } else {
        nbr_nw = MPI_PROC_NULL;
    }
    if (nbr_e != MPI_PROC_NULL && nbr_s != MPI_PROC_NULL) {
        int se_coords[2] = {coords[0] + 1, coords[1] - 1};
        MPI_Cart_rank(cart_comm, se_coords, &nbr_se);
    } else {
        nbr_se = MPI_PROC_NULL;
    }
    if (nbr_w != MPI_PROC_NULL && nbr_s != MPI_PROC_NULL) {
        int sw_coords[2] = {coords[0] - 1, coords[1] - 1};
        MPI_Cart_rank(cart_comm, sw_coords, &nbr_sw);
    } else {
        nbr_sw = MPI_PROC_NULL;
    }

    if (rank == 0) {
        std::cout << "2D Domain Decomposition: " << dims[0] << " × " << dims[1]
                  << " processor grid\n";
        std::cout << "Local domain per rank: " << local_Nx << " × " << local_Ny
                  << "\n";
    }
}

// ============================================================================
// 2D Halo Exchange (4 edges + 4 corners)
// ============================================================================
void Simulation2D::halo_exchange_2d() {
    // ------------------------------------------------------------------
    // D2Q9 direction reference:
    //   0: rest (0,0)   1: E (1,0)    2: N (0,1)    3: W (-1,0)
    //   4: S (0,-1)     5: NE (1,1)   6: NW (-1,1)  7: SW (-1,-1)  8: SE (1,-1)
    //
    // Streaming: f_new(x,y,i) = f(x - cx[i], y - cy[i], i)
    // So at each boundary we need incoming directions from neighbors:
    //   From N: dirs 4,7,8 (southward-moving)
    //   From S: dirs 2,5,6 (northward-moving)
    //   From E: dirs 3,6,7 (westward-moving)
    //   From W: dirs 1,5,8 (eastward-moving)
    //   From NE corner: dir 7
    //   From NW corner: dir 8
    //   From SE corner: dir 6
    //   From SW corner: dir 5
    // ------------------------------------------------------------------
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
    // ---- Pack edge data on device ----

    // North edge: send our top interior row (y = local_Ny), dirs 4,7,8
    Kokkos::parallel_for(
        "pack_n", l_Nx, KOKKOS_LAMBDA(int x) {
            l_send_n(x, 0) = l_f(x + 1, l_Ny, 4);
            l_send_n(x, 1) = l_f(x + 1, l_Ny, 7);
            l_send_n(x, 2) = l_f(x + 1, l_Ny, 8);
        });
    // South edge: send our bottom interior row (y = 1), dirs 2,5,6
    Kokkos::parallel_for(
        "pack_s", l_Nx, KOKKOS_LAMBDA(int x) {
            l_send_s(x, 0) = l_f(x + 1, 1, 2);
            l_send_s(x, 1) = l_f(x + 1, 1, 5);
            l_send_s(x, 2) = l_f(x + 1, 1, 6);
        });
    // East edge: send our right interior column (x = local_Nx), dirs 3,6,7
    Kokkos::parallel_for(
        "pack_e", l_Ny, KOKKOS_LAMBDA(int y) {
            l_send_e(y, 0) = l_f(l_Ny, y + 1, 3);
            l_send_e(y, 1) = l_f(l_Ny, y + 1, 6);
            l_send_e(y, 2) = l_f(l_Ny, y + 1, 7);
        });
    // West edge: send our left interior column (x = 1), dirs 1,5,8
    Kokkos::parallel_for(
        "pack_w", l_Ny, KOKKOS_LAMBDA(int y) {
            l_send_w(y, 0) = l_f(1, y + 1, 1);
            l_send_w(y, 1) = l_f(1, y + 1, 5);
            l_send_w(y, 2) = l_f(1, y + 1, 8);
        });

    // ---- Pack corner data (single values) ----
    // Send to NE: our NE corner cell f(local_Nx, local_Ny, 5)
    // Send to NW: our NW corner cell f(1, local_Ny, 6)
    // Send to SE: our SE corner cell f(local_Nx, 1, 8)
    // Send to SW: our SW corner cell f(1, 1, 7)
    Kokkos::parallel_for(
        "pack_corners", 4, KOKKOS_LAMBDA(int c) {
            switch (c) {
            case 0:
                l_send_corner(c) = l_f(l_Nx, l_Ny, 5);
                break; // → NE
            case 1:
                l_send_corner(c) = l_f(1, l_Ny, 6);
                break; // → NW
            case 2:
                l_send_corner(c) = l_f(l_Nx, 1, 8);
                break; // → SE
            case 3:
                l_send_corner(c) = l_f(1, 1, 7);
                break; // → SW
            }
        });

    Kokkos::fence();

    // ---- Copy to host ----
    Kokkos::deep_copy(h_send_n, send_n);
    Kokkos::deep_copy(h_send_s, send_s);
    Kokkos::deep_copy(h_send_e, send_e);
    Kokkos::deep_copy(h_send_w, send_w);
    Kokkos::deep_copy(h_send_corner, send_corner);

    // ---- MPI Communication ----
    // Tags: 0=N, 1=S, 2=E, 3=W, 4=NE, 5=NW, 6=SE, 7=SW
    // Each process sends to a neighbor and receives from the opposite
    // direction. Send to N, recv from S (tag 0) Send to S, recv from N (tag 1)
    // Send to E, recv from W (tag 2)
    // Send to W, recv from E (tag 3)
    // Corner sends use tags 4-7

    MPI_Request reqs[16];
    int n_req = 0;

    // Edge receives
    MPI_Irecv(h_recv_n.data(), local_Nx * 3, MPI_DOUBLE, nbr_n, 1, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(h_recv_s.data(), local_Nx * 3, MPI_DOUBLE, nbr_s, 0, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(h_recv_e.data(), local_Ny * 3, MPI_DOUBLE, nbr_e, 3, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(h_recv_w.data(), local_Ny * 3, MPI_DOUBLE, nbr_w, 2, cart_comm,
              &reqs[n_req++]);

    // Corner receives
    // Recv from NE (they send us dir 7 → our recv_corner(0))
    // Recv from NW (they send us dir 8 → our recv_corner(1))
    // Recv from SE (they send us dir 6 → our recv_corner(2))
    // Recv from SW (they send us dir 5 → our recv_corner(3))
    MPI_Irecv(&h_recv_corner(0), 1, MPI_DOUBLE, nbr_ne, 5, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(&h_recv_corner(1), 1, MPI_DOUBLE, nbr_nw, 4, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(&h_recv_corner(2), 1, MPI_DOUBLE, nbr_se, 7, cart_comm,
              &reqs[n_req++]);
    MPI_Irecv(&h_recv_corner(3), 1, MPI_DOUBLE, nbr_sw, 6, cart_comm,
              &reqs[n_req++]);

    // Edge sends
    MPI_Isend(h_send_n.data(), local_Nx * 3, MPI_DOUBLE, nbr_n, 0, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(h_send_s.data(), local_Nx * 3, MPI_DOUBLE, nbr_s, 1, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(h_send_e.data(), local_Ny * 3, MPI_DOUBLE, nbr_e, 2, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(h_send_w.data(), local_Ny * 3, MPI_DOUBLE, nbr_w, 3, cart_comm,
              &reqs[n_req++]);

    // Corner sends
    // Send to NE: our dir 5 (tag 4)
    // Send to NW: our dir 6 (tag 5)
    // Send to SE: our dir 8 (tag 6)
    // Send to SW: our dir 7 (tag 7)
    MPI_Isend(&h_send_corner(0), 1, MPI_DOUBLE, nbr_ne, 4, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(&h_send_corner(1), 1, MPI_DOUBLE, nbr_nw, 5, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(&h_send_corner(2), 1, MPI_DOUBLE, nbr_se, 6, cart_comm,
              &reqs[n_req++]);
    MPI_Isend(&h_send_corner(3), 1, MPI_DOUBLE, nbr_sw, 7, cart_comm,
              &reqs[n_req++]);

    MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

    // ---- Copy to device ----
    Kokkos::deep_copy(recv_n, h_recv_n);
    Kokkos::deep_copy(recv_s, h_recv_s);
    Kokkos::deep_copy(recv_e, h_recv_e);
    Kokkos::deep_copy(recv_w, h_recv_w);
    Kokkos::deep_copy(recv_corner, h_recv_corner);

    // ---- Unpack edge data into ghost cells ----

    // North ghost row (y = local_Ny + 1): dirs 4, 7, 8
    Kokkos::parallel_for(
        "unpack_n", local_Nx, KOKKOS_LAMBDA(int x) {
            l_f(x + 1, l_Ny + 1, 4) = l_recv_n(x, 0);
            l_f(x + 1, l_Ny + 1, 7) = l_recv_n(x, 1);
            l_f(x + 1, l_Ny + 1, 8) = l_recv_n(x, 2);
        });
    // South ghost row (y = 0): dirs 2, 5, 6
    Kokkos::parallel_for(
        "unpack_s", local_Nx, KOKKOS_LAMBDA(int x) {
            l_f(x + 1, 0, 2) = l_recv_s(x, 0);
            l_f(x + 1, 0, 5) = l_recv_s(x, 1);
            l_f(x + 1, 0, 6) = l_recv_s(x, 2);
        });
    // East ghost column (x = local_Nx + 1): dirs 3, 6, 7
    Kokkos::parallel_for(
        "unpack_e", local_Ny, KOKKOS_LAMBDA(int y) {
            l_f(l_Nx + 1, y + 1, 3) = l_recv_e(y, 0);
            l_f(l_Nx + 1, y + 1, 6) = l_recv_e(y, 1);
            l_f(l_Nx + 1, y + 1, 7) = l_recv_e(y, 2);
        });
    // West ghost column (x = 0): dirs 1, 5, 8
    Kokkos::parallel_for(
        "unpack_w", local_Ny, KOKKOS_LAMBDA(int y) {
            l_f(0, y + 1, 1) = l_recv_w(y, 0);
            l_f(0, y + 1, 5) = l_recv_w(y, 1);
            l_f(0, y + 1, 8) = l_recv_w(y, 2);
        });

    // ---- Unpack corner data ----
    // NE ghost corner (local_Nx+1, local_Ny+1): dir 7
    // NW ghost corner (0, local_Ny+1):          dir 8
    // SE ghost corner (local_Nx+1, 0):          dir 6
    // SW ghost corner (0, 0):                   dir 5
    Kokkos::parallel_for(
        "unpack_corners", 4, KOKKOS_LAMBDA(int c) {
            switch (c) {
            case 0:
                l_f(l_Nx + 1, l_Ny + 1, 7) = l_recv_corner(c);
                break;
            case 1:
                l_f(0, l_Ny + 1, 8) = l_recv_corner(c);
                break;
            case 2:
                l_f(l_Nx + 1, 0, 6) = l_recv_corner(c);
                break;
            case 3:
                l_f(0, 0, 5) = l_recv_corner(c);
                break;
            }
        });

    Kokkos::fence();
}

// ============================================================================
// Step
// ============================================================================
void Simulation2D::step(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        Kokkos::Timer step_timer;

        halo_exchange_2d();
        double comm_time = step_timer.seconds();
        step_timer.reset();

        // Compute kernels — must use local_Nx, local_Ny, offset_x, offset_y
        // These are modified versions of the existing kernels that account
        // for ghost cells on all four sides and local x-offset.
        streaming_2d(f, f_next, bounce, u_lid, offset_x, offset_y, global_Nx,
                     global_Ny, cell_type, local_Nx, local_Ny);
        std::swap(f, f_next);
        apply_sources_2d(f, cell_type, local_Nx, local_Ny);
        compute_velocity_2d(rho, v, f, local_Nx, local_Ny);
        collide_2d(f, rho, v, omega, local_Nx, local_Ny);
        // Wait for GPU computations to complete!
        Kokkos::fence(); 
        total_compute_time += step_timer.seconds();
        total_comm_time += comm_time;
    }

    // Visualization prep
    velocity_to_speed_2d(local_Nx, local_Ny, v, speed);
    double local_max = get_max_speed_2d(local_Nx, local_Ny, speed, max_speed);
    MPI_Allreduce(&local_max, &max_speed, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    speed_to_rgb_2d(local_Nx, local_Ny, speed, rgb_speed, max_speed);
    velocity_dir_to_rgb_2d(local_Nx, local_Ny, v, rgb_direction, max_speed);
    density_to_rgb_2d(local_Nx, local_Ny, rho, rgb_density);
}

// ============================================================================
// 2D Gather for Visualization
// ============================================================================
void Simulation2D::gather_pixels_2d(PixelField local_rgb,
                                    HostPixelField &global_rgb,
                                    const std::string &name) {
    auto local_host = Kokkos::create_mirror_view(local_rgb);
    Kokkos::deep_copy(local_host, local_rgb);

    if (rank == 0) {
        global_rgb = HostPixelField(name, global_Nx * global_Ny);
    }

    // Two-step gather:
    // Step 1: Gather along rows (x-direction) to the leftmost process in each
    // row Step 2: Gather columns (y-direction) from row leaders to rank 0

    // Create row sub-communicator (processes sharing the same y-coordinate)
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];
    remain_dims[0] = 1; // keep x dimension
    remain_dims[1] = 0; // free y dimension
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

    remain_dims[0] = 0; // free x dimension
    remain_dims[1] = 1; // keep y dimension
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Step 1: Gather within each row → row leader (col_rank == 0 in col_comm,
    //         which corresponds to coords[1] == 0)
    std::vector<uint32_t> row_gathered;
    if (col_rank == 0) {
        row_gathered.resize(local_Nx * row_size * local_Ny);
    }
    MPI_Gather(local_host.data(), local_Nx * local_Ny, MPI_UINT32_T,
               row_gathered.data(), local_Nx * local_Ny, MPI_UINT32_T, 0,
               col_comm); // gather along column → y-direction leader

    // Actually, let me reconsider: we need to gather along x first (row_comm),
    // then along y (col_comm). Let me redo this properly.

    // Step 1: Gather along x within each row
    std::vector<uint32_t> row_full;
    if (row_rank == 0) { // leftmost in row
        row_full.resize(global_Nx * local_Ny);
    }
    MPI_Gather(local_host.data(), local_Nx * local_Ny, MPI_UINT32_T,
               row_full.data(), local_Nx * local_Ny, MPI_UINT32_T, 0, row_comm);

    // Step 2: Gather full rows along y to rank 0
    if (row_rank == 0) {
        MPI_Gather(row_full.data(), global_Nx * local_Ny, MPI_UINT32_T,
                   global_rgb.data(), global_Nx * local_Ny, MPI_UINT32_T, 0,
                   col_comm);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

HostPixelField Simulation2D::get_global_rgb_speed() {
    HostPixelField global_rgb;
    gather_pixels_2d(rgb_speed, global_rgb, "global_rgb_speed_2d");
    return global_rgb;
}

HostPixelField Simulation2D::get_global_rgb_direction() {
    HostPixelField global_rgb;
    gather_pixels_2d(rgb_direction, global_rgb, "global_rgb_dir_2d");
    return global_rgb;
}

HostPixelField Simulation2D::get_global_rgb_density() {
    HostPixelField global_rgb;
    gather_pixels_2d(rgb_density, global_rgb, "global_rgb_dens_2d");
    return global_rgb;
}

// ============================================================================
// Global Reductions
// ============================================================================
double Simulation2D::get_total_density() {
    double local_mass = 0;
    auto l_rho = rho;
    Kokkos::parallel_reduce(
        "local_mass_2d",
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

double Simulation2D::get_total_kinetic_energy() {
    double local_ke = 0;
    auto l_rho = rho;
    auto l_v = v;
    Kokkos::parallel_reduce(
        "local_ke_2d",
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

void Simulation2D::reset() {
    initialize_lbm_2d(global_Nx, global_Ny, local_Nx, local_Ny, offset_x,
                      offset_y, rho, v, f, f_next,
                      InitialisationPattern::Empty);
}