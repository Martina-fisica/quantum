// Variational Monte Carlo for atoms with up to two electrons

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "mc_sampling_cl.h"
#include "E_pot.h"
#include <vector>


using namespace  std;
using namespace arma;


// output file as global variable
ofstream ofile;

// prints to screen the results of the calculations
void  output(double, double, double, int, int, int, vec, vec, mat, mat, int);

// Begin of main program

//int main()
int main(int argc, char* argv[])
{
  char *outfilename;
  int number_cycles, max_variations, charge;
  int dimension, number_particles, J;
  double timestep, omega, stepa, stepb, start;
  vector<Random*> randoms;
  randoms.push_back(new Random(-1));
  randoms.push_back(new Random(-2));
  randoms.push_back(new Random(-3));
  randoms.push_back(new Random(-4));

  // Read in output file, abort if there are too few command-line arguments
  if( argc <= 1 ){
    cout << "Bad Usage: " << argv[0] <<
      " read also output file on same line" << endl;
    exit(1);
  }
  else{
    outfilename=argv[1];
  }
  ofile.open(outfilename);
  //   Read in data
  initialise(stepa, stepb, omega, dimension, start, number_particles, charge,
             max_variations, number_cycles, timestep) ;


  if (number_particles != 2 && number_particles != 6)
  {
      cout <<"I would rather if you inserted 2 or 6 for the number of particles. Start the program again!" <<endl;
      exit(1);
  }
  vec cumulative_e(max_variations+1), cumulative_e2(max_variations+1);
  mat m_cumulative(max_variations +1, max_variations +1);
  mat m_cumulative2(max_variations +1, max_variations +1);
//  mat r_mean2(max_variations +1, max_variations + 1);
//  vec r_mean(max_variations+1);
//  r_mean.zeros();
//  r_mean2.zeros();


  //  Do the mc sampling
  mc_sampling(stepa, stepb, omega, dimension, number_particles, charge,
              max_variations, start,
          number_cycles, timestep, cumulative_e, cumulative_e2, m_cumulative, m_cumulative2, J, randoms);


  // Print out results
  output(start, stepa, stepb, max_variations, number_cycles, charge, cumulative_e, cumulative_e2, m_cumulative, m_cumulative2, J);
  cumulative_e.reset();
  cumulative_e2.reset();
  m_cumulative.reset();
  m_cumulative2.reset();
  //r_mean.reset();
  ofile.close();  // close output file
  return 0;
}



void output(double start, double stepa, double stepb, int max_variations, int number_cycles, int charge,
            vec cumulative_e, vec cumulative_e2, mat m_cumulative, mat m_cumulative2, int J)
{
  int i, j;
  double alpha, beta, variance, error;
  alpha = start;
  for( i=1; i <= max_variations; i++){
    alpha += stepa;
    if(J == 1)
    {
          beta = start;
          for( j=1; j<=max_variations; j++)
          {
              beta  += stepb;
            variance = m_cumulative2(i,j)-m_cumulative(i,j)*m_cumulative(i,j);
            error=sqrt(variance/number_cycles);
            ofile << setiosflags(ios::showpoint | ios::uppercase);
            ofile << setw(15) << setprecision(8) << alpha;
            ofile << setw(15) << setprecision(8) << beta;
            ofile << setw(15) << setprecision(8) << m_cumulative(i,j);
            ofile << setw(15) << setprecision(8) << variance;
            ofile << setw(15) << setprecision(8) << error << endl;
//            if( number_particles == 2)
//            {
//                    ofile << setw(15) << setprecision(8) << r_mean2(i,j) << endl;
//
//            }
          }
    }
          else
    {
    variance = cumulative_e2[i]-cumulative_e[i]*cumulative_e[i];
    error=sqrt(variance/number_cycles);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << alpha;
    ofile << setw(15) << setprecision(8) << cumulative_e[i];
    ofile << setw(15) << setprecision(8) << variance;
    ofile << setw(15) << setprecision(8) << error << endl;
//    if( number_particles == 2)
//    {
//            ofile << setw(15) << setprecision(8) << r_mean(i) << endl;
//    }
    }

//    fprintf(output_file, "%12.5E %12.5E %12.5E %12.5E \n", alpha,cumulative_e[i],variance, error );
  }
//  fclose (output_file);
}  // end of function output







