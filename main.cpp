#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

extern double __enzyme_autodiff(void*, ...);
int enzyme_const, enzyme_dup, enzyme_out;

double square(double x) {
    return x * x;
}
double dsquare(double x) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff((void*) square, x);
}

double loss(double* __restrict__ A, double* __restrict__ B, int n)
{
  double lossVal = 0.0;
  for(int i=0;i<n;i++) 
    lossVal += A[i]*A[i] + B[i]*B[i];
  return lossVal;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Функция с несколькими гауссианами
double f1(double x) {
    return exp(-(x - 2.0) * (x - 2.0) / 2.0) + 0.8 * exp(-(x - 5.0) * (x - 5.0) / 1.0) - 0.5 * exp(-(x - 8.0) * (x - 8.0) / 0.5);
}

// Функция с тремя гауссианами
double f2(double x) {
    return 0.7 * exp(-(x - 1.0) * (x - 1.0) / 1.0) + exp(-(x - 4.0) * (x - 4.0) / 0.5) - 0.4 * exp(-(x - 7.0) * (x - 7.0) / 2.0);
}

// Функция с четырьмя гауссианами
double f3(double x) {
    return 0.6 * exp(-(x - 1.5) * (x - 1.5) / 0.8) + exp(-(x - 4.0) * (x - 4.0) / 0.5) - 0.3 * exp(-(x - 6.5) * (x - 6.5) / 1.5) + 0.4 * exp(-(x - 9.0) * (x - 9.0) / 1.2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() 
{
  for(double i=1; i<5; i++)
    printf("square(%f)=%f, dsquare(%f)=%f\n", i, square(i), i, dsquare(i));

  constexpr int N = 4;
  double A[N] = {1,1,1,1};
  double dA[N] = {0,0,0,0};

  double B[N] = {2,2,2,2};
  double dB[N] = {0,0,0,0};

  __enzyme_autodiff((void*)loss,
                    enzyme_dup, A, dA,
                    enzyme_dup, B, dB,
                    enzyme_const, N);
   
  std::cout << std::endl;
  for(int i=0;i<N;i++)
    std::cout << dA[i] << " ";
  std::cout << std::endl;

  for(int i=0;i<N;i++)
    std::cout << dB[i] << " ";
  std::cout << std::endl;

  std::ofstream fout("data.csv");
  fout << "x;f1;f2;f3;f1xf2xf3" << std::endl;
  
  for(double x = 0.0f; x < 10.0; x+= 0.1)
    fout << x << ";" << f1(x) << ";" << f2(x) << ";" << f3(x) << ";" << f1(x)*f2(x)*f3(x) << ";" << std::endl;
}
