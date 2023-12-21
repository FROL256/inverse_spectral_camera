#pragma once

#include "LiteMath.h"
#include "Image2d.h"
#include <vector>

struct Spectrum
{
  float              Sample(float lambda) const;
  std::vector<float> Resample(int channels);

  std::vector<float> wavelengths; // sorted by wavelength
  std::vector<float> values;      // sorted by wavelength   
};

Spectrum LoadSPDFromFile(const char* path);

constexpr static float LAMBDA_MIN = 360.0f;
constexpr static float LAMBDA_MAX = 830.0f;

std::vector<float> Get_CIE_lambda();
std::vector<float> Get_CIE_X();
std::vector<float> Get_CIE_Y();
std::vector<float> Get_CIE_Z();

std::vector<float> LoadAndResampleSpectrum(const char* path, int channels);
std::vector<float> LoadAndResampleAllCheckerSpectrum(const char* folder_path, int channels);

// 2° standard colorimetric observer
inline LiteMath::float3 XYZToRGB(LiteMath::float3 xyz)
{
  LiteMath::float3 rgb;
  rgb[0] = +3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
  rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
  rgb[2] = +0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
  return rgb;
}


static inline LiteMath::float3 SpectrumToXYZ(float spec, float lambda, float lambda_min, float lambda_max,
                                             const float* a_CIE_X, const float* a_CIE_Y, const float* a_CIE_Z) 
{
  const float pdf = 1.0f / (lambda_max - lambda_min);
  const float CIE_Y_integral = 106.856895f;
  const uint32_t nCIESamples = 471;
  
  spec = (pdf != 0) ? spec / pdf : 0.0f;

  float X,Y,Z; 
  {
    uint32_t offset = uint32_t(float(std::floor(lambda + 0.5f)) - lambda_min);
  
    if (offset >= nCIESamples)
      X = 0;
    else
      X = a_CIE_X[offset];
  
    if (offset >= nCIESamples)
      Y = 0;
    else
      Y = a_CIE_Y[offset];
  
    if (offset >= nCIESamples)
      Z = 0;
    else
      Z = a_CIE_Z[offset];
  }

  {
    X *= spec;
    Y *= spec;
    Z *= spec;
  }

  float x = X / CIE_Y_integral;
  float y = Y / CIE_Y_integral;
  float z = Z / CIE_Y_integral;

  return LiteMath::float3{x ,y, z};
}