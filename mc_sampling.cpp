#include <cmath>
#include <omp.h>
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
#include "Random.h"
#include <omp.h>

using namespace  std;
using namespace arma;

// the step length and its squared inverse for the second derivative
#define num_core 2
#define h 0.001
#define h2 1000000

void quantum_force(mat&, mat&, double, double, double, int, int, double, int);

// Monte Carlo sampling with the Metropolis algorithm

void mc_sampling(double stepa, double stepb, double omega, int dimension, int number_particles, int charge,
                 int max_variations, double starta, double startb,
                 int number_cycles, double timestep,
                 vec& r_mean, mat& r_mean2,
                 vec& e_kv, vec& e_potv, mat& e_km, mat& e_potm,
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
        #pragma omp parallel for num_threads(num_core)
        for (int variate=1; variate <= max_variations; variate++){
            mat r_old, r_new, qforce_old, qforce_new;
            r_old = zeros<mat>(number_particles, dimension);
            r_new = zeros<mat>(number_particles, dimension);
            qforce_old = zeros<mat>(number_particles, dimension);
            qforce_new = zeros<mat>(number_particles, dimension);

            double e_k, e_pot;

            // initialisations of variational parameters and energies
            double alpha = starta + variate*stepa;
            double energy= 0; double delta_e=0; double energy2=0;
            double beta=0;
            //  initial trial position, note calling with alpha
            //  and in three dimensions
            for (int i = 0; i < number_particles; i++) {
                for (int j=0; j < dimension; j++) {
                    r_old(i,j) = randoms[omp_get_thread_num()]->nextGauss() *sqrt(timestep);
                }
            }

            double wfold = wave_function(omega, r_old, alpha, beta, dimension, number_particles, J);
            quantum_force (r_old, qforce_old, alpha, beta, wfold, number_particles, dimension, omega, J);


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

                    double wfnew = wave_function(omega, r_new, alpha, beta, dimension, number_particles, J);
                    quantum_force(r_new, qforce_new, alpha, beta, wfnew, number_particles, dimension, omega, J);

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
                }

                // compute local energy
                    delta_e = local_energy(e_k, e_pot, omega, r_old, el, alpha, beta, wfold, dimension,
                                           number_particles, charge, J);
                    e_kv(variate) += e_k;
                    e_potv(variate) += e_pot;
                    // update energies
                    energy += delta_e;
                    energy2 += delta_e*delta_e;
            }   // end of loop over MC trials



            cout << "variational parameter= " << alpha <<endl;
            // update the energy average and its squared
            e_kv(variate) = e_kv(variate)/number_cycles;
            e_potv(variate) = e_potv(variate)/number_cycles;
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

                r_mean(variate) = sqrt(r_m);
            }

        }
    }// end of loop over variational  steps



    else if(J == 1)
    {
        // loop over variational parameters
         #pragma omp parallel for num_threads(num_core)
        for (int variate=1; variate <= max_variations; variate++){
            mat r_old, r_new, qforce_old, qforce_new;
            r_old = zeros<mat>(number_particles, dimension);
            r_new = zeros<mat>(number_particles, dimension);
            qforce_old = zeros<mat>(number_particles, dimension);
            qforce_new = zeros<mat>(number_particles, dimension);

            double e_k, e_pot;

            // initialisations of variational parameters and energies
            double alpha = starta + variate*stepa;
            double energy= 0; double delta_e=0; double energy2=0;
            // initialisations of variational parameters and energies
            #pragma omp parallel for num_threads(num_core)
            for(int variate2 =1; variate2 <= max_variations; variate2 ++)
            {
                delta_e=0;
                energy = energy2 = 0;
                double beta = startb +variate2*stepb;

                //  initial trial position, note calling with alpha
                //  and in three dimensions
                for (int i = 0; i < number_particles; i++) {
                    for (int j=0; j < dimension; j++) {
                        r_old(i,j) = randoms[omp_get_thread_num()]->nextGauss() *sqrt(timestep);
                    }
                }

                double wfold = wave_function(omega, r_old, alpha, beta, dimension, number_particles, J);
                quantum_force (r_old, qforce_old, alpha, beta, wfold, number_particles, dimension, omega, J);

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



                       double wfnew = wave_function(omega, r_new, alpha, beta, dimension, number_particles, J);
                        quantum_force(r_new, qforce_new, alpha, beta, wfnew, number_particles, dimension, omega, J);

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
                    }

                    // compute local energy
                        delta_e = local_energy(e_k, e_pot, omega, r_old, el, alpha, beta, wfold, dimension,
                                               number_particles, charge, J);
                        e_km(variate, variate2) += e_k;
                        e_potm(variate, variate2) += e_pot;
                        // update energies
                        energy += delta_e;
                        energy2 += delta_e*delta_e;

                }   // end of loop over MC trials

                cout << "variational parameter= " << alpha <<endl;
                cout << "variational parameter2= " << beta <<endl;
                // update the energy average and its squared
                e_km(variate, variate2) = e_km(variate, variate2)/number_cycles;
                e_potm(variate, variate2) = e_potm(variate, variate2)/number_cycles;
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

                    r_mean2(variate, variate2) = sqrt(r_m);
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

// Function to calculate the local energy with num derivative

double  local_energy(double& e_k, double& e_pot, double omega, mat& r, int el, double alpha, double beta, double wfold, int dimension,
                     int number_particles, int charge, int J)
{
    int i, j;
    double e_local, wfminus, wfplus, e_kinetic, e_potential,
            r_single_particle;
    mat r_plus;
    mat r_minus;
    mat r_12(number_particles, number_particles);
    // allocate matrices which contain the position of the particles
    r_plus = r_minus = r;
    r_12.zeros();

    distance(r, r_12, number_particles, dimension);

    // compute the kinetic energy. it should fit well for any kind of wave function
    e_kinetic = 0;
    for (i = 0; i < number_particles; i++) {
        for (j = 0; j < dimension; j++) {
            r_plus(i,j) = r(i,j)+h;
            r_minus(i,j) = r(i,j)-h;
            wfminus = wave_function(omega, r_minus, alpha, beta, dimension, number_particles, J);
            wfplus  = wave_function(omega, r_plus, alpha, beta, dimension, number_particles, J);
            e_kinetic -= (wfminus+wfplus-2.0*wfold);
            r_plus(i,j) = r(i,j);
            r_minus(i,j) = r(i,j);
        }
    }


    // include electron mass and hbar squared and divide by wave function
    //cout <<"wfold = " <<wfold <<endl;
    e_kinetic = 0.5*h2*e_kinetic/wfold;

    e_k = e_kinetic;


    // compute the potential energy
  //  if(pot == 0)
  //  {
  //      E_pot potenziale1(r, dimension, number_particles, charge, r_12, r_single_particle);
  //      e_potential = potenziale1.atom();
  //  }
        E_pot potenziale2(omega, r, dimension, number_particles, charge, r_12, r_single_particle);
        e_potential = potenziale2.oscillator();

    // contribution from electron-electron potential

    if(el == 2)
    {
        E_pot potenziale3(omega, r, dimension, number_particles, charge, r_12, r_single_particle);
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

    e_pot = e_potential;


    r_plus.reset(); //free memory
    r_minus.reset();

    e_local = e_potential+e_kinetic;
    return e_local;
}

void initialise(double& stepa, double& stepb, double& omega, int& dimension, double& starta, double& startb, int& number_particles, int& charge,
                int& max_variations, int& number_cycles, double& timestep)
{
    cout << "number of particles = ";
    cin >> number_particles;
    cout << "charge of nucleus = ";
    cin >> charge;
    cout << "dimensionality = ";
    cin >> dimension;
    cout <<"start alpha=";
    cin >> starta;
    cout <<"start beta=";
    cin >> startb;
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

void quantum_force(mat& r, mat& qforce, double alpha, double beta, double wf, int number_particles, int dimension, double omega, int J)
{
    int i, j;
    double wfminus, wfplus;
    mat r_plus(number_particles, dimension), r_minus(number_particles, dimension);
    for(i = 0; i < number_particles; i++)
    {
        for(j = 0; j < dimension; j++)
        {
            r_plus(i,j) = r_minus(i,j) = r(i,j);
        }
    }

    // compute the f i r s t derivative
    for(i = 0; i < number_particles; i++)
    {
        for(j = 0; j < dimension; j++)
        {
            r_plus(i,j) = r(i,j) + h;
            r_minus(i,j) = r(i,j) - h;
            wfminus = wave_function(omega, r_minus, alpha, beta, dimension, number_particles, J);
            wfplus = wave_function(omega, r_plus, alpha, beta, dimension, number_particles, J);
            qforce(i,j) =  (wfplus - wfminus)*2.0/wf/(2*h);
            r_plus(i,j) = r(i,j);
            r_minus(i,j) = r(i,j);
        }
    }
} //end of function quantum_force




