#pragma once

#include "LiteMath.h"
#include "Image2d.h"
#include <vector>

struct Spectrum
{
  float Sample(float lambda) const;
  std::vector<float> wavelengths; // sorted by wavelength
  std::vector<float> values;      // sorted by wavelength   
};

Spectrum LoadSPDFromFile(const char* path);

constexpr static float LAMBDA_MIN = 360.0f;
constexpr static float LAMBDA_MAX = 830.0f;

std::vector<float> LoadAndResampleSpectrum(const char* path, int channels);
std::vector<float> LoadAndResampleAllCheckerSpectrum(const char* folder_path, int channels);