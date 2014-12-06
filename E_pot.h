#include <iostream>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <armadillo>
//#include "lib.h"

using namespace std;
using namespace arma;


#ifndef E_POT_H
#define E_POT_H

class E_pot
{
public:
    // member variables
    double  m_r_single_particle, m_dimension, m_number_particles, m_charge, m_omega;
    mat m_r;
    mat m_r_12;
    // constructor
    E_pot(double omega, mat& r, int dimension, int number_particles, int charge, mat& r_12, double r_single_particle);
    // specific function
    double atom(), electronelectron(), oscillator();
};

#endif // E_POT_H
