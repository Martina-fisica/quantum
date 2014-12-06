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

void mc_sampling(double, double, double, int, int, int, int, double, int, double, vec &, vec &, mat&, mat&, int &, vector<Random*> &randoms);

// Function to read in data from screen, note call by reference
void initialise(double&, double&, double&, int&, double&, int&, int&,
                int&, int&, double&) ;



// The local energy
double  local_energy(int, mat&, mat&, vec& , mat&, vec&, mat&, int, int, int, int,int );

void distance(mat&, mat&, int, int );

void quantum_force(mat&, mat&, mat&, int, int);

void pieces(mat&, int, double, int, int, mat&, mat&, vec&, vec&, double, mat&, mat&, mat&, mat&);
