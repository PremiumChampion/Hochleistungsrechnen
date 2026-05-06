#include <Kokkos_Core.hpp>

// Define alias types for cleaner code
// We use compile-time dimensions ([2] and [9]) for the last indices.
using ScalarField2D = Kokkos::View<double **>;
using VectorField2D = Kokkos::View<double **[2]>;
using DistFuncD2Q9 = Kokkos::View<double **[9]>;

void initialize_lbm(int Nx, int Ny, ScalarField2D &rho, VectorField2D &v,
                    DistFuncD2Q9 &f, DistFuncD2Q9 &f_next);
void compute_velocity(ScalarField2D rho, VectorField2D v, DistFuncD2Q9 f);
void streaming(DistFuncD2Q9 f_in, DistFuncD2Q9 f_out);