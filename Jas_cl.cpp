#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "mc_sampling.h"
#include "E_pot.h"
#include "wave.h"

using namespace  std;
using namespace arma;

double Jas(mat& a, mat &r_12, double beta, int dimension, int number_particles, int J)
{
    if(J == 0)
        return 1;

    int i, j;

    double n2 = number_particles/2.0;
    double Jas=1;


    for (i = 0; i < number_particles; i++)
    {
        for (j = i + 1; j < number_particles; j++)
        {
            if (number_particles == 2 | number_particles == 6)
            Jas *= exp(a(i,j) * r_12(i,j) / (1.0 + beta * r_12(i,j)));
        }
    }




    return Jas;

}
