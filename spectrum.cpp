#include "spectrum.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using LiteMath::lerp;
using LiteMath::clamp;

inline size_t BinarySearch(const float* array, size_t array_sz, float val) 
{
  int32_t last = (int32_t)array_sz - 2, first = 1;
  while (last > 0) 
  {
    size_t half = (size_t)last >> 1, 
    middle = first + half;
    bool predResult = array[middle] <= val;
    first = predResult ? int32_t(middle + 1) : first;
    last = predResult ? last - int32_t(half + 1) : int32_t(half);
  }
  return (size_t)clamp(int32_t(first - 1), 0, int32_t(array_sz - 2));
}

float Spectrum::Sample(float lambda) const
{
  if (wavelengths.empty() || lambda < wavelengths.front() || lambda > wavelengths.back())
    return 0;
 
  // int o = BinarySearch(wavelengths.size(), [&](int i) { return wavelengths[i] <= lambda; });
  int o = BinarySearch(wavelengths.data(), wavelengths.size(), lambda);

  float t = (lambda - wavelengths[o]) / (wavelengths[o + 1] - wavelengths[o]);
  return lerp(values[o], values[o + 1], t);
}


Spectrum LoadSPDFromFile(const char* path)
{
  Spectrum res;
  std::ifstream in(path);
  std::string line;
  while(std::getline(in, line))
  {
    if(line.size() > 0 && line[0] == '#')
      continue;

    auto pos = line.find_first_of(' ');
    float lambda = std::stof(line.substr(0, pos));
    float spec   = std::stof(line.substr(pos + 1, line.size() - 1));
    res.wavelengths.push_back(lambda);
    res.values.push_back(spec);
  }

  return res;
}

std::vector<float> LoadAndResampleSpectrum(const char* path, int channels)
{
  Spectrum specData = LoadSPDFromFile(path);

  std::vector<float> res(channels);
  std::fill(res.begin(), res.end(), 0.0f);
  
  const float step = 1.0f;
  for(int c=0;c<channels;c++) {
    float lambdaStart = LAMBDA_MIN + (float(c+0)/float(channels))*(LAMBDA_MAX - LAMBDA_MIN);
    float lambdaEnd   = LAMBDA_MIN + (float(c+1)/float(channels))*(LAMBDA_MAX - LAMBDA_MIN);
    
    float summ = 0.0f;
    int numSamples = 0;
    for(float lambda = lambdaStart; lambda <= lambdaEnd; lambda += step) {
      summ += specData.Sample(lambda);
      numSamples++;
    }
    res[c] = summ / float(numSamples);
  }

  return res;
}

std::vector<float> LoadAndResampleAllCheckerSpectrum(const char* folder_path, int channels)
{
  std::string folderPath(folder_path);
  std::vector<std::string> allPaths = {"H1", "H2", "H3", "H4", "H5", "H6",
                                       "G1", "G2", "G3", "G4", "G5", "G6",
                                       "F1", "F2", "F3", "F4", "F5", "F6",
                                       "E1", "E2", "E3", "E4", "E5", "E6",};

  std::vector<float> allData;
  allData.reserve(allPaths.size()*channels);

  for(auto path : allPaths) {
    std::string fullPath = folderPath + "/" + path + ".spd";
    //std::cout << "fullPath = " << fullPath.c_str() << std::endl;
    auto spec = LoadAndResampleSpectrum(fullPath.c_str(), channels);
    allData.insert(allData.end(), spec.begin(), spec.end());
  }

  return allData;
}