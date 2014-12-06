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


#ifndef DET_H
#define DET_H

class Det
{
public:
    // member variables
    double  m_omega, m_alpha;
    mat m_r;
    // constructor
    Det(mat& r, double omega, double alpha);
    // specific function
    double det2(), det6();
};

#endif // E_POT_H

double Slater(mat&, mat&, mat&, int, int, double, double);
