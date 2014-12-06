int index1, index2, dim;
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
psi1 = inv(psi1);
psi2 = inv(psi2);
mat r_12(number_particles, number_particles);
r_12.zeros();
distance(r, r_12, number_particles, dimension);


for(index1=0; index1<number_particles; index1++)
{
    for(dim=0; dim<dimension; dim++)
    {
        for(index2=0; index2<number_particles; index2++)
    {
       if(index2 =! index1)
          piece1(index1, dim) += a(index1,index2)*(r(i,dim)-r(index1,dim))/(r_12(index1,index2)*pow((1+beta*r_12(index1,index2)),2));
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
        if( index2=! index1)
        piece2(index1) += (a(index1,index2)*(dimension-3)*(beta*r_12(index1,index2)+1)+2)/(r_12(index1,index2)*pow((1+beta*r_12(index1,index2)),3));
}

if(i%2 == 0)
{
    for(index1=0; index1<number_particles/2; index1++)
    {
            piece3(index1,0) = -pow(omega,4)*r(index1,0)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(0,index1)
                          -(pow(omega,2)*r(index1,0)-1)*(pow(omega,2)*r(index1,0)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(1,index1)
                          -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(2,index1);
            piece3(index1,1) = -pow(omega,4)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(0,index1)
                          -pow(omega,4)*r(index1,0)*r(index1,1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(1,index1)
                          -(pow(omega,2)*r(index1,1)-1)*(pow(omega,2)*r(index1,1)+1)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(2,index1);
            piece4(index1) = pow(omega,4)*(pow(omega,2)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-2)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(0,index1)
                        +pow(omega,4)*r(index1,0)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(1,index1)
                        +pow(omega,4)*r(index1,1)*(pow(omega,4)*(r(index1,0)*r(index1,0)+r(index1,1)*r(index1,1))-4)*exp(-0.5*pow(omega,4)*pow(r(index1,0),2))*psi2(2,index1);
    }
}
    else
    {
        for(index1=0; index1<number_particles/2; index1++)
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
