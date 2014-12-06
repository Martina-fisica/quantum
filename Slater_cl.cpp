#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
#include "Slater.h"
#include "Hermite1.h"

using namespace  std;
using namespace arma;

double Slater(mat& psi1, mat& psi2, mat& r, int dimension, int number_particles, double omega, double alpha)
{
    //Construction of wave functions
    int number_particles_2 =number_particles/2;
    mat r1(number_particles_2, dimension), r2(number_particles_2, dimension);
    r1.zeros();
    r2.zeros();

    int i, j, k, q_num_tot=0, count = 0;
    vec q_num(dimension);
    q_num.zeros();

    // splitting of the matrix of the positions
    for(i=0; i<number_particles_2; i++)
        for(j=0; j<dimension; j++)
            r1(i,j) = r(i,j);
    for(i=number_particles_2; i<number_particles; i++)
        for(j=0; j<dimension; j++)
            r2(i-number_particles_2, j) = r(i,j);


    int index;


    for(i=0; i<number_particles_2; i++)
    {
        if(count >= q_num_tot+1)
        {
            count = 0;
            q_num_tot++;
        }
        q_num(0) = q_num_tot - count;
        q_num(1) = count;

        for(j=0; j<number_particles_2; j++)
        {
            psi1(i,j) = 1;
            psi2(i,j) = 1;

            index=0;

            for(k=0; k<dimension; k++)
            {

                psi1(i,j) *= Hermite(r1(j,k)*sqrt(omega), q_num(k))*exp(-alpha*omega*r1(i,k)*r1(i,k)/2);
                psi2(i,j) *= Hermite(r2(j,k)*sqrt(omega), q_num(k))*exp(-alpha*omega*r2(i,k)*r2(i,k)/2);
            }


        }

        count ++;

    }


    double det1, det2;
    if (number_particles == 2)
    {
        Det result(r, omega, alpha);
        det1 = result.det2();
            return det1;
    }
     else if(number_particles == 6)
    {
        Det result1(psi1, omega, alpha);
        Det result2(psi2, omega, alpha);
        det1 = result1.det6();
        det2 = result2.det6();
        return det1*det2;
    }
  /* else if(number_particles == 12)
    {
        Det result(r, dimension, number_particles);
        det = result.det12();
        return det;
    }*/
}


double Det::det6()
{
    return det(m_r);
}

double Det::det2()
{
    double r1, r2;
    r1 = sqrt(pow(m_r(0,0), 2.0) + pow(m_r(0,1), 2.0));
    r2 = sqrt(pow(m_r(1,0), 2.0) + pow(m_r(1,1), 2.0));
  return exp(-m_alpha*m_omega*(r1*r1 + r2*r2)/2);
}

Det::Det(mat& r, double omega, double alpha)
{
    m_r = r;
    m_alpha = alpha;
    m_omega = omega;

}
