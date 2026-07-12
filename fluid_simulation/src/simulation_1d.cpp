#include "simulation_1d.h"
#include <iostream>

Simulation1D::Simulation1D(int _Nx, int _Ny, double _omega,
                           InitialisationPattern pattern, bool _bounce)
    : global_Nx(_Nx),
      omega(_omega),
      bounce(_bounce) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_Ny = _Ny / size;
    global_Ny = local_Ny * size;
    offset_y = rank * local_Ny;

    // Fields: full x, local y + 2 ghost rows
    speed = ScalarField2D("speed", global_Nx, local_Ny + 2);
    rgb_speed = PixelField("rgb_spd", global_Nx * local_Ny);
    rgb_direction = PixelField("rgb_dir", global_Nx * local_Ny);
    rgb_density = PixelField("rgb_dens", global_Nx * local_Ny);
    cell_type = CellTypeField("cells", global_Nx, local_Ny + 2);
    h_cell_type = Kokkos::create_mirror_view(cell_type);
    Kokkos::deep_copy(cell_type, 0);

    // Halo buffers: 3 D2Q9 directions per edge cell
    send_top = HaloBuf("send_top", global_Nx, 3);
    recv_top = HaloBuf("recv_top", global_Nx, 3);
    send_bottom = HaloBuf("send_bottom", global_Nx, 3);
    recv_bottom = HaloBuf("recv_bottom", global_Nx, 3);

    h_send_top = Kokkos::create_mirror_view(send_top);
    h_recv_top = Kokkos::create_mirror_view(recv_top);
    h_send_bottom = Kokkos::create_mirror_view(send_bottom);
    h_recv_bottom = Kokkos::create_mirror_view(recv_bottom);

    initialize_lbm(global_Nx, global_Ny, local_Ny, offset_y, rho, v, f, f_next,
                   pattern);
}

void Simulation1D::step(size_t num_iterations) {
    for (size_t i = 0; i < num_iterations; i++) {
        Kokkos::Timer step_timer;
        auto l_Ny = this->local_Ny;
        auto l_send_top = this->send_top;
        auto l_send_bottom = this->send_bottom;
        auto l_recv_top = this->recv_top;
        auto l_recv_bottom = this->recv_bottom;
        auto l_f = this->f;
        auto l_f_next = this->f_next;

        // ---- Halo exchange (y-direction only) ----
        Kokkos::parallel_for(
            "pack_top", global_Nx, KOKKOS_LAMBDA(int x) {
                l_send_top(x, 0) = l_f(x, l_Ny, 4); // S
                l_send_top(x, 1) = l_f(x, l_Ny, 7); // SW
                l_send_top(x, 2) = l_f(x, l_Ny, 8); // SE
            });
        Kokkos::parallel_for(
            "pack_bottom", global_Nx, KOKKOS_LAMBDA(int x) {
                l_send_bottom(x, 0) = l_f(x, 1, 2); // N
                l_send_bottom(x, 1) = l_f(x, 1, 5); // NE
                l_send_bottom(x, 2) = l_f(x, 1, 6); // NW
            });
        Kokkos::fence();
        Kokkos::deep_copy(h_send_top, send_top);
        Kokkos::deep_copy(h_send_bottom, send_bottom);

        int top_neighbor = (rank + 1) % size;
        int bottom_neighbor = (rank - 1 + size) % size;
        if (bounce) {
            if (rank == 0)
                bottom_neighbor = MPI_PROC_NULL;
            if (rank == size - 1)
                top_neighbor = MPI_PROC_NULL;
        }

        MPI_Request reqs[4];
        MPI_Irecv(h_recv_top.data(), global_Nx * 3, MPI_DOUBLE, top_neighbor, 0,
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_recv_bottom.data(), global_Nx * 3, MPI_DOUBLE,
                  bottom_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(h_send_top.data(), global_Nx * 3, MPI_DOUBLE, top_neighbor, 1,
                  MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(h_send_bottom.data(), global_Nx * 3, MPI_DOUBLE,
                  bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        Kokkos::deep_copy(recv_top, h_recv_top);
        Kokkos::deep_copy(recv_bottom, h_recv_bottom);

        Kokkos::parallel_for(
            "unpack_top", global_Nx, KOKKOS_LAMBDA(int x) {
                l_f(x, l_Ny + 1, 4) = l_recv_top(x, 0);
                l_f(x, l_Ny + 1, 7) = l_recv_top(x, 1);
                l_f(x, l_Ny + 1, 8) = l_recv_top(x, 2);
            });
        Kokkos::parallel_for(
            "unpack_bottom", global_Nx, KOKKOS_LAMBDA(int x) {
                l_f(x, 0, 2) = l_recv_bottom(x, 0);
                l_f(x, 0, 5) = l_recv_bottom(x, 1);
                l_f(x, 0, 6) = l_recv_bottom(x, 2);
            });
        Kokkos::fence();

        double comm_time = step_timer.seconds();
        step_timer.reset();

        // ---- Compute kernels (unchanged from existing code) ----
        streaming(f, f_next, bounce, u_lid, offset_y, global_Ny, cell_type);
        std::swap(f, f_next);
        apply_sources(f, cell_type, local_Ny);
        compute_velocity(rho, v, f, local_Ny);
        collide(f, rho, v, omega, local_Ny);

        // Wait for GPU computations to complete!
        Kokkos::fence();

        total_compute_time += step_timer.seconds();
        total_comm_time += comm_time;
    }

    // Post-step visualization prep
    velocity_to_speed(global_Nx, local_Ny, v, speed);
    double local_max = get_max_speed(global_Nx, local_Ny, speed, max_speed);
    // double step_max = 0.0;
    // MPI_Allreduce(&local_max, &step_max, 1, MPI_DOUBLE, MPI_MAX,
    //               MPI_COMM_WORLD);
    // max_speed = std::max(max_speed, step_max); // Never let the scale drop
    speed_to_rgb(global_Nx, local_Ny, speed, rgb_speed, max_speed);
    velocity_dir_to_rgb(global_Nx, local_Ny, v, rgb_direction, max_speed);
    density_to_rgb(global_Nx, local_Ny, rho, rgb_density);
}

// Gather functions — identical to existing simulation class
HostPixelField Simulation1D::get_global_rgb_speed() {
    HostPixelField global_rgb;
    if (rank == 0)
        global_rgb = HostPixelField("global_rgb_speed", global_Nx * global_Ny);
    auto local_rgb_host = Kokkos::create_mirror_view(rgb_speed);
    Kokkos::deep_copy(local_rgb_host, rgb_speed);
    MPI_Gather(local_rgb_host.data(), local_rgb_host.extent(0), MPI_UINT32_T,
               global_rgb.data(), local_rgb_host.extent(0), MPI_UINT32_T, 0,
               MPI_COMM_WORLD);
    return global_rgb;
}

void Simulation1D::reset() {
    initialize_lbm(global_Nx, global_Ny, local_Ny, offset_y, rho, v, f, f_next,
                   InitialisationPattern::Empty);
}

double Simulation1D::get_total_density() {
    double local_mass = 0;
    auto l_rho = rho; // Alias view inside scope to prevent "this" illegal
                      // access on Device lambda
    Kokkos::parallel_reduce(
        "local_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1},
                                               {global_Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            sum += l_rho(x, y);
        },
        local_mass);

    double global_mass = 0;
    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_mass;
}

double Simulation1D::get_total_kinetic_energy() {
    double local_ke = 0;
    auto l_rho = rho;
    auto l_v = v;
    Kokkos::parallel_reduce(
        "local_ke",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1},
                                               {global_Nx, local_Ny + 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            double vx = l_v(x, y, 0);
            double vy = l_v(x, y, 1);
            sum += 0.5 * l_rho(x, y) * (vx * vx + vy * vy);
        },
        local_ke);

    double global_ke = 0;
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_ke;
}

HostVectorField2D Simulation1D::get_global_velocity() {
    HostVectorField2D global_v;
    if (rank == 0) {
        global_v = HostVectorField2D("global_v", global_Nx, global_Ny);
    }

    auto local_v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(local_v_host, v);

    std::vector<double> send_buf(global_Nx * local_Ny * 2);
    for (int y = 1; y <= local_Ny; ++y) {
        for (int x = 0; x < global_Nx; ++x) {
            send_buf[((y - 1) * global_Nx + x) * 2 + 0] = local_v_host(x, y, 0);
            send_buf[((y - 1) * global_Nx + x) * 2 + 1] = local_v_host(x, y, 1);
        }
    }

    std::vector<double> recv_buf;
    if (rank == 0)
        recv_buf.resize(global_Nx * global_Ny * 2);

    MPI_Gather(send_buf.data(), send_buf.size(), MPI_DOUBLE, recv_buf.data(),
               send_buf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int y = 0; y < global_Ny; ++y) {
            for (int x = 0; x < global_Nx; ++x) {
                global_v(x, y, 0) = recv_buf[(y * global_Nx + x) * 2 + 0];
                global_v(x, y, 1) = recv_buf[(y * global_Nx + x) * 2 + 1];
            }
        }
    }
    return global_v;
}

HostPixelField Simulation1D::get_global_rgb_direction() {
    HostPixelField global_rgb;
    if (rank == 0) {
        global_rgb = HostPixelField("global_rgb_dir", global_Nx * global_Ny);
    }

    auto local_rgb_host = Kokkos::create_mirror_view(rgb_direction);
    Kokkos::deep_copy(local_rgb_host, rgb_direction);

    MPI_Gather(local_rgb_host.data(), local_rgb_host.extent(0), MPI_UINT32_T,
               global_rgb.data(), local_rgb_host.extent(0), MPI_UINT32_T, 0,
               MPI_COMM_WORLD);

    return global_rgb;
}

HostPixelField Simulation1D::get_global_rgb_density() {
    HostPixelField global_rgb;
    if (rank == 0)
        global_rgb = HostPixelField("global_rgb_dens", global_Nx * global_Ny);

    auto local_rgb_host = Kokkos::create_mirror_view(rgb_density);
    Kokkos::deep_copy(local_rgb_host, rgb_density);

    MPI_Gather(local_rgb_host.data(), local_rgb_host.extent(0), MPI_UINT32_T,
               global_rgb.data(), local_rgb_host.extent(0), MPI_UINT32_T, 0,
               MPI_COMM_WORLD);

    return global_rgb;
}
