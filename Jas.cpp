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

double Jas(mat &r_12, double beta, int dimension, int number_particles, int J)
{
    if(J == 0)
        return 1;

    int i, j;
    double a_sym, a_asym;
    double Jas=1;
    mat a;
    a = zeros<mat>(number_particles, number_particles);

    if (dimension == 2)
    {
        a_sym = 1./3.;
        a_asym = 1.0;
    } else if (dimension == 3) {
        a_sym = 1. / 4.;
        a_asym = 1. / 2.;
    } else {
        cout << "Unable to initialize Jastrow paremters: Unknown dimension" <<endl;
        exit(1);
    }

    for (i = 0; i < number_particles; i++) {
        for (j = 0; j < number_particles; j++) {

            if (i%2 == j%2) {
                a(i,j) = a_sym;
            } else {
                a(i,j) = a_asym;
            }
        }
    }


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
