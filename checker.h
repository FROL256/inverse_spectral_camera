#pragma once

#include "LiteMath.h"
#include "Image2d.h"
#include <vector>

using LiteMath::int2;
using LiteMath::int3;

using LiteMath::float2;
using LiteMath::float3;

struct Rect
{
  int2 bMin;
  int2 bMax;
};

std::vector<Rect>   GetCheckerRects();
std::vector<float3> LoadAveragedCheckerLDRData(const char* path, const std::vector<Rect>& a_rectData); 
std::vector<float>  AveragedSpectrumFromImage3D(const float* data, int width, int height, int channels, const std::vector<Rect>& a_rectData);
std::vector<float3> AveragedColor4f(const float* data, int width, int heaight, const std::vector<Rect>& a_rectData);
