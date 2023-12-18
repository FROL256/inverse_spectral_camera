#include "checker.h"

using LiteMath::uchar4;

std::vector<Rect> GetCheckerRects()
{
  std::vector<Rect> res; res.reserve(24);
  
  int2 size (60,60);

  res.push_back({{496,324},{560,390}});
  res.push_back({{590,325},int2(590,325) + size});
  res.push_back({{688,328},int2(688,328) + size});
  res.push_back({{786,329},int2(786,329) + size});
  res.push_back({{883,330},int2(883,330) + size});
  res.push_back({{981,332},int2(981,332) + size});

  for(int y=1;y<4;y++) {
    for(int x=0;x<6;x++) {
      auto rectOld = res[(y-1)*6 + x];
      rectOld.bMin.y += 96;
      rectOld.bMax.y += 96;
      res.push_back(rectOld);
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