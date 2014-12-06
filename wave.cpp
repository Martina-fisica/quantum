#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "mc_sampling.h"
#include "Jas.h"
#include "Slater.h"
//#include "E_pot.h"

using namespace  std;
using namespace arma;

// Function to compute the squared wave function, simplest form

double  wave_function(double omega, mat &r, double alpha, double beta, int dimension, int number_particles, int J)
{
    mat r_12;
    r_12 = zeros<mat>(number_particles, number_particles);
    double jas, slater;

    distance(r, r_12, number_particles, dimension);

    jas = Jas(r_12, beta, dimension, number_particles, J);

    slater = Slater(r, dimension, number_particles, omega, alpha); // omega = 1 only for the moment!!!

    return jas*slater;

}
