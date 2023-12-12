#include <stdio.h>
#include <iostream>

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

}
