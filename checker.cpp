#include "checker.h"
#include <iostream>
#include <fstream>

using LiteMath::uchar4;

std::vector<Rect> GetCheckerRects()
{
  std::vector<Rect> res; 
  res.reserve(24);

  int X[6] = {525, 624, 720, 817, 915, 1015};
  int Y[4] = {345, 445, 545, 640};
  const int halfSize = 20;

  for(int y=0;y<4;y++) {
    for(int x=0;x<6;x++) {
      Rect rect;
      rect.bMin.x = X[x] - halfSize;
      rect.bMin.y = Y[y] - halfSize;
      rect.bMax.x = X[x] + halfSize;
      rect.bMax.y = Y[y] + halfSize;
      res.push_back(rect);
    }
  }
  return res;
}


std::vector<float3> LoadAveragedCheckerLDRData(const char* path, const std::vector<Rect>& a_rectData)
{
  LiteImage::Image2D<uchar4> image = LiteImage::LoadImage<uchar4>(path, 1.0f);
  
  std::vector<float3> res(a_rectData.size());
  for(size_t rectId = 0; rectId < a_rectData.size(); rectId++) 
  { 
    auto rect = a_rectData[rectId];
    float3 summ(0,0,0);
    int pixelNum = 0;
    for(int y=rect.bMin.y; y<rect.bMax.y;y++) {
      for(int x = rect.bMin.x; x<rect.bMax.x;x++) {
        uchar4 pixel = image[int2(x,y)];
        summ.x += float(pixel.x);
        summ.y += float(pixel.y);
        summ.z += float(pixel.z);
        pixelNum++;
      }
    }
    summ /= float(pixelNum);
    res[rectId] = summ; // / 255
  }

  return res;
} 

std::vector<float> LoadAveragedSpectrumFromImage3d1f(const char* path, const std::vector<Rect>& a_rectData, int* pChannels)
{
  std::ifstream fin(path, std::ios::binary);
  int xyz[3] = {};
  fin.read((char*)xyz, sizeof(int)*3);

  const int width    = xyz[0];
  const int height   = xyz[1];
  const int channels = xyz[2];

  (*pChannels) = channels;

  std::vector<float> data(width*height*channels);
  fin.read((char*)data.data(), sizeof(float)*data.size());
  fin.close();
  
  std::vector<float> allSpecters(channels*a_rectData.size());

  for(int c=0;c<xyz[2];c++)
  {
    float* imData = data.data() + width*height*c;
    for(size_t rectId = 0; rectId < a_rectData.size(); rectId++) 
    { 
      auto rect = a_rectData[rectId];
      float summ = 0.0f;
      int pixelNum = 0;
      for(int y=rect.bMin.y; y<rect.bMax.y;y++) {
        for(int x = rect.bMin.x; x<rect.bMax.x;x++) {
          float pixel = imData[y*width+x];
          summ += pixel;
          pixelNum++;
        }
      }
      summ /= float(pixelNum);
      //if(rectId == 12)
      //  std::cout << summ << std::endl;
      allSpecters[rectId*channels + c] = summ; 
    }

  }

  return allSpecters;
}