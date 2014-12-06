#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "mc_sampling_cl.h"
#include "Jas_cl.h"
#include "Slater_cl.h"
//#include "E_pot.h"

using namespace  std;
using namespace arma;

// Function to compute the squared wave function, simplest form

double  wave_function(mat& psi1, mat& psi2, mat& a, double omega, mat &r, double alpha, double beta, int dimension, int number_particles, int J)
{
    double jas, slater;
    mat r_12;
    r_12 = zeros<mat>(number_particles, number_particles);

    distance(r, r_12, number_particles, dimension);

    jas = Jas(a, r_12, beta, dimension, number_particles, J);

    slater = Slater(psi1, psi2, r, dimension, number_particles, omega, alpha);

    return jas*slater;

}
