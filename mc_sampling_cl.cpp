#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "mc_sampling_cl.h"
#include "E_pot.h"
#include "wave_cl.h"
#include "Random.h"
#include <omp.h>

using namespace  std;
using namespace arma;

// the step length and its squared inverse for the second derivative
#define h 0.001
#define h2 1000000

// Monte Carlo sampling with the Metropolis algorithm

void mc_sampling(double stepa, double stepb, double omega, int dimension, int number_particles, int charge,
                 int max_variations, double start,
                 int number_cycles, double timestep,
                 vec& cumulative_e, vec& cumulative_e2, mat& m_cumulative, mat& m_cumulative2, int& J, vector<Random*> &randoms)
{

    double D = 0.5;

    int el;
    cout <<"If you want to consider the electron-electron repulsion type 2; if you don't type 3:" <<endl;
    cin >> el;

    J=0;
    cout <<"Do you want the Jastrow factor (1 for yes, 0 for no)?:" <<endl;
    cin >> J;

    if(J == 0)
    {
        // loop over variational parameters
#pragma omp parallel for num_threads(2)
        for (int variate=1; variate <= max_variations; variate++){
            mat a;
            a = zeros<mat>(number_particles, number_particles);
            mat psi1(number_particles/2, number_particles/2);
            psi1.zeros();
            mat psi2(number_particles/2, number_particles/2);
            psi2.zeros();
            vec piece2(number_particles), piece4(number_particles);
            mat piece1(number_particles, dimension), piece3(number_particles, dimension);
            piece1.zeros();
            piece2.zeros();
            piece3.zeros();
            piece4.zeros();
            mat r_12(number_particles, number_particles);
            r_12.zeros();
            mat r_old, r_new, qforce_old, qforce_new;
            r_old = zeros<mat>(number_particles, dimension);
            r_new = zeros<mat>(number_particles, dimension);
            qforce_old = zeros<mat>(number_particles, dimension);
            qforce_new = zeros<mat>(number_particles, dimension);
            // initialisations of variational parameters and energies
            double alpha = start + variate*stepa;
            double energy= 0; double delta_e=0; double energy2=0;
            double beta=0;
            //  initial trial position, note calling with alpha
            //  and in three dimensions
            for (int i = 0; i < number_particles; i++) {
                for (int j=0; j < dimension; j++) {
                    r_old(i,j) = randoms[omp_get_thread_num()]->nextGauss() *sqrt(timestep);
                }
            }

            distance(r_old, r_12, number_particles, dimension);
            double wfold = wave_function(psi1, psi2, a, omega, r_old, alpha, beta, dimension, number_particles, J);
            pieces(r_12, dimension, beta, J, number_particles, piece1, piece3, piece2, piece4, omega, a, psi1, psi2, r_old);
            quantum_force(piece1, piece3, qforce_old, number_particles, dimension);

            // loop over monte carlo cycles

            for (int cycles = 1; cycles <= number_cycles; cycles++){
                // new position

                for (int i = 0; i < number_particles; i++) {

                    for (int j=0; j < dimension; j++) {
                        r_new(i,j) = r_old(i,j) + randoms[omp_get_thread_num()]->nextGauss()*sqrt(timestep) + qforce_old(i,j)*timestep*D;
                    }

                    for(int k = 0 ; k < number_particles; k++)
                    {
                        if( k != i)
                        {
                            for(int j = 0; j < dimension; j++)
                            {
                                r_new(k,j) = r_old(k,j);
                            }
                        }
                    }

                    distance(r_new, r_12, number_particles, dimension);
                    double wfnew = wave_function(psi1, psi2, a, omega, r_new, alpha, beta, dimension, number_particles, J);
                    pieces(r_12, dimension, beta, J, number_particles, piece1, piece3, piece2, piece4, omega, a, psi1, psi2, r_new);
                    quantum_force(piece1, piece3, qforce_new, number_particles, dimension);


                    // we compute the log of the ratio of the greens functions to be used in the Metropolis_Hastings algorithm
                    double greensfunction = 0.0;
                    for(int j = 0; j < dimension; j++)
                    {
                        greensfunction += 0.5*(qforce_old(i,j) + qforce_new(i,j))*(D*timestep*0.5*(qforce_old(i,j) - qforce_new(i,j)) -
                                                                                   r_new(i,j) + r_old(i,j));
                    }
                    greensfunction = exp(greensfunction);

                    // The Metropolis test is performed by moving one particle at the time
                    if(randoms[omp_get_thread_num()]->nextDouble() <= greensfunction*wfnew*wfnew/wfold/wfold)
                    {
                        for(int j=0; j < dimension; j++)
                        {
                            r_old(i,j) = r_new(i,j);
                            qforce_old(i,j) = qforce_new(i,j);
                        }
                        wfold = wfnew;
                    }

                    distance(r_old, r_12, number_particles, dimension);
                    // compute local energy
                    delta_e += local_energy(i, r_12, piece1, piece2, piece3, piece4, r_old, el, dimension, number_particles, charge, J);
                }

                delta_e = 0.5*delta_e;
                // update energies
                energy += delta_e;
                energy2 += delta_e*delta_e;
            }   // end of loop over MC trials



            cout << "variational parameter= " << alpha <<endl;
            // update the energy average and its squared
            cumulative_e(variate) = energy/number_cycles;
            cumulative_e2(variate) = energy2/number_cycles;

            if(number_particles == 2)
            {
                double r_m = 0;
                for (int i = 0; i < number_particles-1; i++) {
                    for (int j = i+1; j < number_particles; j++) {
                        for (int k = 0; k < dimension; k++) {
                            r_m += (r_old(i,k)-r_old(j,k))*(r_old(i,k)-r_old(j,k));
                        }
                    }
                }

                //r_mean(variate) = sqrt(r_m);
            }

        }
    }// end of loop over variational  steps



    else if(J == 1)
    {

        //creation of the matrix that takes spins into account
        double a_sym, a_asym;
        mat a;
        a = zeros<mat>(number_particles, number_particles);

        if (dimension == 2)
        {
            a_sym = 1./ 3;
            a_asym = 1.0;
        } else if (dimension == 3) {
            a_sym = 1. / 4;
            a_asym = 1. / 2;
        } else {
            cout << "Unable to initialize Jastrow paremters: Unknown dimension" <<endl;
            exit(1);
        }

        for (int i = 0; i < number_particles-1; i++) {
            for (int j = 0; j < number_particles-1; j++) {

                if ( j%2==i%2 ) {
                    a(i,j) = a_sym;
                } else {
                    a(i,j) = a_asym;
                }
            }
        }



        // loop over variational parameters
#pragma omp parallel for num_threads(2)
        for (int variate=1; variate <= max_variations; variate++){
            mat psi1(number_particles/2, number_particles/2);
            psi1.zeros();
            mat psi2(number_particles/2, number_particles/2);
            psi2.zeros();
            vec piece2(number_particles), piece4(number_particles);
            mat piece1(number_particles, dimension), piece3(number_particles, dimension);
            piece1.zeros();
            piece2.zeros();
            piece3.zeros();
            piece4.zeros();
            mat r_12(number_particles, number_particles);
            r_12.zeros();
            mat r_old, r_new, qforce_old, qforce_new;
            r_old = zeros<mat>(number_particles, dimension);
            r_new = zeros<mat>(number_particles, dimension);
            qforce_old = zeros<mat>(number_particles, dimension);
            qforce_new = zeros<mat>(number_particles, dimension);
            // initialisations of variational parameters and energies
            double alpha = start + variate*stepa;
            double energy= 0; double delta_e=0; double energy2=0;
            // initialisations of variational parameters and energies
#pragma omp parallel for num_threads(2)
            for(int variate2 =1; variate2 <= max_variations; variate2 ++)
            {
                delta_e=0;
                energy = energy2 = 0;
                double beta = start +variate2*stepb;

                //  initial trial position, note calling with alpha
                //  and in three dimensions
                for (int i = 0; i < number_particles; i++) {
                    for (int j=0; j < dimension; j++) {
                        r_old(i,j) = randoms[omp_get_thread_num()]->nextGauss() *sqrt(timestep);
                    }
                }

                distance(r_old, r_12, number_particles, dimension);
                double wfold = wave_function(psi1, psi2, a, omega, r_old, alpha, beta, dimension, number_particles, J);
                pieces(r_12, dimension, beta, J, number_particles, piece1, piece3, piece2, piece4, omega, a, psi1, psi2, r_old);                
                quantum_force(piece1, piece3, qforce_old, number_particles, dimension);

                // loop over monte carlo cycles
                for (int cycles = 1; cycles <= number_cycles; cycles++){
                    // new position
                    for (int i = 0; i < number_particles; i++) {

                        for (int j=0; j < dimension; j++) {
                            r_new(i,j) = r_old(i,j) + randoms[omp_get_thread_num()]->nextGauss()*sqrt(timestep) + qforce_old(i,j)*timestep*D;
                        }
                        for(int k = 0 ; k < number_particles; k++)
                        {
                            if( k != i)
                            {
                                for(int j = 0; j < dimension; j++)
                                {
                                    r_new(k,j) = r_old(k,j);
                                }
                            }
                        }

                        distance(r_new, r_12, number_particles, dimension);
                        double wfnew = wave_function(psi1, psi2, a, omega, r_new, alpha, beta, dimension, number_particles, J);
                        pieces(r_12, dimension, beta, J, number_particles, piece1, piece3, piece2, piece4, omega, a, psi1, psi2, r_new);
                        quantum_force(piece1, piece3, qforce_new, number_particles, dimension);

                        // we compute the log of the ratio of the greens functions to be used in the Metropolis_Hastings algorithm
                        double greensfunction = 0.0;
                        for(int j = 0; j < dimension; j++)
                        {
                            greensfunction += 0.5*(qforce_old(i,j) + qforce_new(i,j))*(D*timestep*0.5*(qforce_old(i,j) - qforce_new(i,j)) -
                                                                                       r_new(i,j) + r_old(i,j));
                        }
                        greensfunction = exp(greensfunction);

                        // The Metropolis test is performed by moving one particle at the time
                        if(randoms[omp_get_thread_num()]->nextDouble() <= greensfunction*wfnew*wfnew/wfold/wfold)
                        {
                            for(int j=0; j < dimension; j++)
                            {
                                r_old(i,j) = r_new(i,j);
                                qforce_old(i,j) = qforce_new(i,j);
                            }
                            wfold = wfnew;
                        }

                        distance(r_old, r_12, number_particles, dimension);
                        // compute local energy
                        delta_e += local_energy(i, r_12, piece1, piece2, piece3, piece4, r_old, el, dimension, number_particles, charge, J);
                    }

                    delta_e = 0.5*delta_e;
                    // update energies
                    energy += delta_e;
                    energy2 += delta_e*delta_e;

                }   // end of loop over MC trials

                cout << "variational parameter= " << alpha <<endl;
                cout << "variational parameter2= " << beta <<endl;
                // update the energy average and its squared
                m_cumulative(variate, variate2) = energy/number_cycles;
                m_cumulative2(variate, variate2) = energy2/number_cycles;

                if(number_particles == 2)
                {
                    double r_m = 0;
                    for (int i = 0; i < number_particles-1; i++) {
                        for (int j = i+1; j < number_particles; j++) {
                            for (int k = 0; k < dimension; k++) {
                                r_m += (r_old(i,k)-r_old(j,k))*(r_old(i,k)-r_old(j,k));
                            }
                        }
                    }

                    //r_mean2(variate, variate2) = sqrt(r_m);
                }

            }

        }
    }

}   // end mc_sampling function


//Distance between particles

void distance(mat& r, mat& r_12, int number_particles, int dimension)
{
    for (int i = 0; i < number_particles; i++) {
        for (int j = i+1; j < number_particles; j++) {
            r_12(i,j) = 0;
            for (int k = 0; k < dimension; k++) {
                r_12(i,j) += (r(i,k)-r(j,k))*(r(i,k)-r(j,k));
            }
            r_12(i,j)= sqrt(r_12(i,j));
            r_12(j,i) = r_12(i,j);

        }
    }
}

// Function to calculate the local energy without num derivative

double  local_energy(int i, mat& r_12, mat& piece1, vec& piece2, mat& piece3, vec& piece4, mat& r, int el, int dimension,
                     int number_particles, int charge, int J)
{

    double e_local, e_kinetic=0, e_potential, r_single_particle;

    // compute the kinetic energy. it should fit well for any kind of wave function

        e_kinetic += piece4(i);
        if(J==1)
        {
            for(int k=0; k<dimension; k++)
            {

                e_kinetic += piece3(i,k)*piece1(i,k);
            }
        }

       // cout <<e_kinetic <<endl;


    // if (iteration%10000==0) cout <<"e_k = " <<e_kinetic <<endl;


    // compute the potential energy
    //  if(pot == 0)
    //  {
    //      E_pot potenziale1(r, dimension, number_particles, charge, r_12, r_single_particle);
    //      e_potential = potenziale1.atom();
    //  }
    E_pot potenziale2(r, dimension, number_particles, charge, r_12, r_single_particle);
    e_potential = potenziale2.oscillator();

    // contribution from electron-electron potential

    if(el == 2)
    {
        E_pot potenziale3(r, dimension, number_particles, charge, r_12, r_single_particle);
        e_potential += potenziale3.electronelectron();
    }
    if(el == 3)
    {
        e_potential += 0;
    }
    if(el != 2 && el!= 3)
    {
        cout <<"Wrong value inserted" <<endl;
        exit(0);
    }

    e_local = e_potential+e_kinetic;
    return e_local;
}

void initialise(double& stepa, double& stepb, double& omega, int& dimension, double& start, int& number_particles, int& charge,
                int& max_variations, int& number_cycles, double& timestep)
{
    cout << "number of particles = ";
    cin >> number_particles;
    cout << "charge of nucleus = ";
    cin >> charge;
    cout << "dimensionality = ";
    cin >> dimension;
    cout <<"start alpha and beta=";
    cin >> start;
    cout << "maximum variational parameters = ";
    cin >> max_variations;
    cout << "# MC steps= ";
    cin >> number_cycles;
    cout << "timestep= ";
    cin >> timestep;
    cout <<"omega= ";
    cin >> omega;
    cout <<"step alpha= ";
    cin >> stepa;
    cout <<"step beta= ";
    cin >> stepb;
}  // end of function initialise

void quantum_force(mat& piece1, mat& piece3, mat& qforce, int number_particles, int dimension)
{
    int i, j;

    for(i=0; i<number_particles; i++)
        for(j=0; j<dimension; j++)
        {
            qforce(i,j) = 2*(piece3(i,j) + piece1(i,j));
        }

} //end of function quantum_force

//calculation of the "pieces" for the analytical derivatives

void pieces(mat& r_12, int dimension, double beta, int J, int number_particles, mat& piece1, mat& piece3, vec& piece2, vec& piece4, double omega, mat& a, mat& psi1, mat& psi2, mat& r)
{
    int index1, index2, dim;
    psi1 = inv(psi1);
    psi2 = inv(psi2);
    if(J==1)
    {
        for(index1=0; index1<number_particles; index1++)
        {
            for(dim=0; dim<dimension; dim++)
            {
                for(index2=0; index2<number_particles; index2++)
                {
                    if(index2 != index1)
                        piece1(index1, dim) += a(index1,index2)*(r(index1,dim)-r(index1,dim))/(r_12(index1,index2)*pow((1+beta*r_12(index1,index2)),2));
                }
            }
        }

        for(index1=0; index1<number_particles; index1++)
        {
            double norm;
            for(dim=0; dim<dimension; dim++)
                norm += piece1(index1,dim)*piece1(index1,dim);
            piece2(index1) += norm;
            for(index2=0; index2<number_particles; index2++)
                if( index2 != index1)
                    piece2(index1) += (a(index1,index2)*(dimension-3)*(beta*r_12(index1,index2)+1)+2)/(r_12(index1,index2)*pow((1+beta*r_12(index1,index2)),3));
        }

        for(index1=0; index1<number_particles/2; index1++)
        {
            if(index1%2 == 0)
            {
                piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1)
                        -(pow(omega,2)*r(index1,0)-1)*(pow(omega,2)*r(index1,0)+1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(2,index1);
                piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1,index1)
                        -(pow(omega,2)*r(index1,1)-1)*(pow(omega,2)*r(index1,1)+1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(2,index1);
                piece4(index1) = pow(omega,4)*(pow(omega,2)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1)
                        +pow(omega,4)*r(index1,0)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1,index1)
                        +pow(omega,4)*r(index1,1)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(2,index1);
            }
        else
        {
                piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(0,index1)
                        -(pow(omega,2)*r(index1,0)-1)*(pow(omega,2)*r(index1,0)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(1,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(2,index1);
                piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(0,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(1,index1)
                        -(pow(omega,2)*r(index1,1)-1)*(pow(omega,2)*r(index1,1)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(2,index1);
                piece4(index1) = pow(omega,4)*(pow(omega,2)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(0,index1)
                        +pow(omega,4)*r(index1,0)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(1,index1)
                        +pow(omega,4)*r(index1,1)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(2,index1);
            }
        }
    }
    else if(J==0)
    {
        for(index1=0; index1<number_particles/2; index1++)
        {
            if(index1%2 == 0)
            {
                if(number_particles==6)
                {
                piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1)
                        -(pow(omega,2)*r(index1,0)-1)*(pow(omega,2)*r(index1,0)+1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(2,index1);
                                    cout <<"psi1, indici impossibili" <<psi1(2, index1) <<endl;
                piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1)
                        -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1,index1)
                         -(pow(omega,2)*r(index1,1)-1)*(pow(omega,2)*r(index1,1)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(2,index1);
                piece4(index1) = pow(omega,4)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0, index1)
                        +pow(omega,4)*r(index1,0)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(1, index1)
                        +pow(omega,4)*r(index1,1)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(2, index1);
                }
                if(number_particles ==2)
                {
                    piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1);
                    piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0,index1);
                    piece4(index1) = pow(omega,4)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi2(0, index1);
                }
            }
            else
            {
                if(number_particles ==6)
                {

                    piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(0,index1)
                            -(pow(omega,2)*r(index1,0)-1)*(pow(omega,2)*r(index1,0)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(1,index1)
                            -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(2,index1);
                    piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(0,index1)
                            -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(1,index1)
                            -(pow(omega,2)*r(index1,1)-1)*(pow(omega,2)*r(index1,1)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi1(2,index1);
                    piece4(index1) = pow(omega,4)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(0, index1)
                        +pow(omega,4)*r(index1,0)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(1, index1)
                        +pow(omega,4)*r(index1,1)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(2, index1);
                }
                if(number_particles==2)
                {
                    piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(0,index1);
                    piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(0,index1);
                    piece4(index1) = pow(omega,4)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*(pow(r(index1,0),2)+pow(r(index1,1),2)))*psi1(0, index1);
                }

        }
        }
    }
} //end of pieces

