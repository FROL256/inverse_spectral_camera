#pragma once
#include <vector>
#include <cmath>

template<typename T> T AdamEps() { return T(1e-8); }
template<> double      AdamEps() { return double(1e-20f); }

template<typename T>
class IGradientOptimizer
{
public:
  virtual ~IGradientOptimizer(){};
  virtual void step(T *params_ptr, const T* grad_ptr, int iter = 0) = 0;
protected:
  IGradientOptimizer() {}
};

template<typename T>
class OptimizerGD : public IGradientOptimizer<T>
{
public:
  OptimizerGD(int a_params_count, float _lr = 0.01f) : params_count(a_params_count) { lr = _lr; }
  virtual void step(T *params_ptr, const T* grad_ptr) override {
    for (size_t i = 0; i< params_count; i++)
      params_ptr[i] -= lr * grad_ptr[i];
  }
private:
  size_t params_count = 0;
  T lr;
};

template<typename T>
struct AdamOptimizer : public IGradientOptimizer<T>
{
  AdamOptimizer(size_t a_size)
  {
    momentum.resize(a_size);
    m_GSquare.resize(a_size);
    std::fill(momentum.begin(), momentum.end(), 0.0);
    std::fill(m_GSquare.begin(), m_GSquare.end(), 0.0);
  }
  
  void step(T* a_state, const T* a_grad, int iter) override
  {
    int factorGamma = iter/100 + 1;
    const T alpha   = 0.5;
    const T beta    = 0.25;
    const T gamma   = 0.25/factorGamma;
    
    // Adam: m[i] = b*mPrev[i] + (1-b)*gradF[i], 
    // GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*grad[i]*grad[i])
    for(size_t i=0;i<momentum.size();i++)
    {
      auto gradVal = a_grad[i];
      momentum [i] = momentum[i]*beta + gradVal*(T(1.0)-beta);
      m_GSquare[i] = T(2.0)*(m_GSquare[i]*alpha + (gradVal*gradVal)*(T(1.0)-alpha)); // does not works without 2.0
    }

    //xNext[i] = x[i] - gamma/(sqrt(GSquare[i] + epsilon)); 
    for (int i=0;i<momentum.size();i++) 
      a_state[i] -= (gamma*momentum[i]/(std::sqrt(m_GSquare[i] + epsilon)));  
  }

  std::vector<T> momentum; 
  std::vector<T> m_GSquare;
  T epsilon = AdamEps<T>();
};

template<typename T>
class AdamOptimizer2 : public IGradientOptimizer<T> // this implementation does not don't works ... 
{
public:
  AdamOptimizer2(int _params_count, T _lr = T(0.01f), T _beta_1 = T(0.9f), T _beta_2 = T(0.999f), T _eps = T(1e-8)) 
  {
    lr = _lr;
    beta_1 = _beta_1;
    beta_2 = _beta_2;
    eps = _eps;
    V = std::vector<T>(_params_count, 0);
    S = std::vector<T>(_params_count, 0);
    iter = 0;
  }
  
  void step(T *params_ptr, const T* grad_ptr, int a_iter) override
  {
    for (size_t i = 0; i < params_count; i++)
    {
      T g = grad_ptr[i];
      V[i] = beta_1 * V[i] + (1 - beta_1) * g;
      T Vh = V[i] / (1 - std::pow(beta_1, iter + 1));
      S[i] = beta_2 * S[i] + (1 - beta_2) * g * g;
      T Sh = S[i] / (1 - std::pow(beta_2, iter + 1));
      params_ptr[i] -= lr * Vh / (std::sqrt(Sh) + eps);
    }
    iter++;
  }
private:
  T lr, beta_1, beta_2, eps;
  int iter = 0;
  std::vector<T> V;
  std::vector<T> S;
  size_t params_count;
};


