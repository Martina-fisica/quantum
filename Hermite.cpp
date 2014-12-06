#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "Hermite.h"

using namespace  std;
using namespace arma;

double Hermite(mat &r, int i, int q_num_x, int q_num_y)
{
    if (q_num_x == 0 && q_num_y == 0)
    {
        return 1;
    }
    else if (q_num_x == 1 && q_num_y==0)
    {
        return 2*r(i,0);
    }
     else if (q_num_y == 1 && q_num_x==0)
    {
        return 2*r(i,1);
    }
 /*   else if(q_num == 2)
    {
        return 4*r(i,j)*r(i,j)-2;
    } */
}
