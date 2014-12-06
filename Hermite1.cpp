#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
//#include "lib.h"
#include "Hermite1.h"

using namespace  std;
using namespace arma;

double Hermite(double coor, int q_num)
{
    if (q_num == 0)
    {
        return 1;
    }
    else if (q_num == 1)
    {
        return 2*coor;
    }
 /*   else if(q_num == 2)
    {
        return 4*r(i,j)*r(i,j)-2;
    } */
}
