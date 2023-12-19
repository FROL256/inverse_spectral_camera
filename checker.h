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
std::vector<float>  LoadAveragedSpectrumFromImage3d1f(const char* path, const std::vector<Rect>& a_rectData, int* pChannels);