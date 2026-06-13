#include "field.h"

#ifndef SIMULATION_H
#define SIMULATION_H

class simulation {
  public:
    int Nx = 1500;
    int Ny = 1000;
    bool bounce = true;
    double omega = 1.0; // 0 to 2
    double max_speed = 0;
    double u_lid = 0.0;
    ScalarField2D rho;
    VectorField2D v;
    ScalarField2D speed;
    DistFuncD2Q9 f;
    DistFuncD2Q9 f_next;
    PixelField rgb_speed;
    PixelField rgb_direction;

  public:
    simulation(uint Nx = 1500, uint Ny = 1000, double omega = 1.0,
               InitialisationPattern pattern = InitialisationPattern::Droplet,
               bool bounce = true);
    ~simulation() = default;
    void step(size_t num_iterations = 1);
};

#endif