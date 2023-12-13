#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

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

struct TestData
{
  void Init(int a_size)
  {
    x_data.resize(a_size); 
    f1_data.resize(a_size);
    f2_data.resize(a_size);
    f3_data.resize(a_size);
    ref_data.resize(a_size);
    
    double x = 0.0;
    double step = 10/double(a_size);
    for(int i=0;i<a_size;i++,x+=step)
    {
      x_data  [i] = x;
      f1_data [i] = 1.0f;
      f2_data [i] = f2(x);
      f3_data [i] = f3(x);
      ref_data[i] = f1(x)*f2(x)*f3(x);
    }
  }

  std::vector<double> x_data; 
  std::vector<double> f1_data;
  std::vector<double> f2_data;
  std::vector<double> f3_data;
  std::vector<double> ref_data; // 
};

double TestDataLoss(double* __restrict__ f1_val, 
                    double* __restrict__ f2_val,
                    double* __restrict__ f3_val,
                    double* __restrict__ ref_val,
                    double* __restrict__ x_data, size_t n)
{
  double lossVal = 0.0;
  for(size_t i=0;i<n;i++) 
  { 
    double diff = f1_val[i]*f2_val[i]*f3_val[i] - ref_val[i];
    lossVal += diff*diff;
  }
  
  return lossVal; 
}

struct AdamOptimizer
{
  std::vector<double> momentum; 
  std::vector<double> grad;
  std::vector<double> m_GSquare;
  
  void Init(int a_size)
  {
    momentum.resize(a_size);
    grad.resize(a_size);
    m_GSquare.resize(a_size);
    std::fill(momentum.begin(), momentum.end(), 0.0);
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(m_GSquare.begin(), m_GSquare.end(), 0.0);
  }

  void UpdateState(double* a_state, int iter)
  {
    int factorGamma    = iter/100 + 1;
    const double alpha = 0.5;
    const double beta  = 0.25;
    const double gamma = 0.25/double(factorGamma);
    
    // Adam: m[i] = b*mPrev[i] + (1-b)*gradF[i], 
    // GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*grad[i]*grad[i])
    for(size_t i=0;i<grad.size();i++)
    {
      momentum [i] = momentum[i]*beta + grad[i]*(1.0-beta);
      m_GSquare[i] = 2.0*(m_GSquare[i]*alpha + (grad[i]*grad[i])*(1.0-alpha)); // does not works without 2.0
    }

    //xNext[i] = x[i] - gamma/(sqrt(GSquare[i] + epsilon)); 
    for (int i=0;i<grad.size();i++) 
      a_state[i] -= (gamma*momentum[i]/(std::sqrt(m_GSquare[i] + double(1e-25f))));  
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() 
{
  TestData data;
  AdamOptimizer opt;

  data.Init(100);  
  opt.Init(100);

  auto initial_f1 = data.f1_data;
  
  for(int iter = 0; iter < 2000; iter++) 
  {
    std::fill(opt.grad.begin(), opt.grad.end(), 0.0);  

    double dloss = __enzyme_autodiff((void*)TestDataLoss,
                                     enzyme_dup,   data.f1_data.data(), opt.grad.data(),
                                     enzyme_const, data.f2_data.data(),
                                     enzyme_const, data.f3_data.data(),
                                     enzyme_const, data.ref_data.data(),
                                     enzyme_const, data.x_data.data(),
                                     enzyme_const, opt.grad.size());
    
    double lossVal = TestDataLoss(data.f1_data.data(),
                                  data.f2_data.data(),
                                  data.f3_data.data(),
                                  data.ref_data.data(),
                                  data.x_data.data(),
                                  opt.grad.size());                                     

    //opt.UpdateGrad();
    opt.UpdateState(data.f1_data.data(), iter);
    
    if((iter+1) % 10 == 0)
      std::cout << "iter = " << iter << ", loss = (" << lossVal << ")" << std::endl;
  }

  std::ofstream fout2("data.csv");
  fout2 << "x;f1;f2;f3;f1xf2xf3;initial_f1;optimized_f1;" << std::endl;

  double x = 0.0;
  double step = 10/double(100);
  for(int i=0;i<100;i++,x+=step)
    fout2 << x << ";" << f1(x) << ";" << f2(x) << ";" << f3(x) << ";" << f1(x)*f2(x)*f3(x) << ";" << initial_f1[i] << ";" << data.f1_data[i] << ";" << std::endl;
}
