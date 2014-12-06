#include <iostream>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <armadillo>
#include "lib.h"
#include <vector>
#include "Random.h"

using namespace std;
using namespace arma;

void mc_sampling(double, double, double, int, int, int,
                 int, double, double,
                 int, double,
                 vec&, mat&,
                 vec&, vec&, mat&, mat&,
                 vec&, vec&, mat&, mat&, int&, vector<Random*> &randoms);

// Function to read in data from screen, note call by reference
void initialise(double &, double&, double&, int&, double &, double &, int&, int&, int&, int&, double&) ;



// The local energy
double  local_energy(double&, double&, double, mat&, int, double, double , double, int, int, int, int);

void distance(mat&, mat&, int, int );


