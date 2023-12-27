#include "checker.h"
#include <iostream>
#include <fstream>

using LiteMath::uchar4;
using LiteMath::int2;

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

using Pixel = int2;

// Функция для проверки допустимости координаты
bool isValid(int x, int y, int rows, int cols) {
    return (x >= 0 && x < rows && y >= 0 && y < cols);
}

// Функция для выполнения обхода в глубину и поиска связанных пикселей
void dfs(const std::vector<uint8_t>& image, int x, int y, int rows, int cols, std::vector<bool>& visited, std::vector<Pixel>& connectedSet) {
    static const int dx[] = { -1, 0, 1, 0 }; // Смещение для соседних пикселей по горизонтали
    static const int dy[] = { 0, 1, 0, -1 }; // Смещение для соседних пикселей по вертикали

    visited[x * cols + y] = true;
    connectedSet.push_back(Pixel(x, y));

    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (isValid(nx, ny, rows, cols) && !visited[nx * cols + ny] && image[nx * cols + ny] == 1) {
            dfs(image, nx, ny, rows, cols, visited, connectedSet);
        }
    }
}

// Функция для поиска связанных множеств на бинарном изображении-маске
std::vector<std::vector<Pixel>> findConnectedSets(const std::vector<uint8_t>& image, int rows, int cols) {
    std::vector<bool> visited(rows * cols, false);
    std::vector<std::vector<Pixel>> connectedSets;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (image[i * cols + j] == 1 && !visited[i * cols + j]) {
                std::vector<Pixel> connectedSet;
                dfs(image, i, j, rows, cols, visited, connectedSet);
                connectedSets.push_back(connectedSet);
            }
        }
    }

    return connectedSets;
}

// Функция для нахождения геометрического центра связанного множества
Pixel findGeometricCenter(const std::vector<Pixel>& connectedSet) {
    int sumX = 0;
    int sumY = 0;

    for (const auto& pixel : connectedSet) {
        sumX += pixel.x;
        sumY += pixel.y;
    }

    int centerX = sumX / connectedSet.size();
    int centerY = sumY / connectedSet.size();

    return Pixel(centerX, centerY);
}

std::vector<Rect> GetCheckerRectFromMask(const char* maskPath)
{
  LiteImage::Image2D<uchar4> imageTemp = LiteImage::LoadImage<uchar4>(maskPath, 1.0f);
  
  int w = imageTemp.width();
  int h = imageTemp.height();
  std::vector<uint8_t> image(w*h);
  for(int y=0;y<h;y++)
    for(int x=0;x<w;x++)
       image[y*w+x] = (imageTemp[int2(x,y)].x == 0) ? 0 : 1;

  auto pixelSets = findConnectedSets(image, h, w);
  std::vector<Pixel> cetners(pixelSets.size());
  for(size_t i=0;i<cetners.size();i++)
    cetners[i] = findGeometricCenter(pixelSets[i]);

  //for(size_t i=0;i<cetners.size();i++)
  //  std::cout << "center[" << i << "] = (" << cetners[i].x << ", " << cetners[i].y << ")" << std::endl;
  //std::cout << "rects num = " << pixelSets.size() << std::endl;
  
  std::vector<Rect> res(cetners.size());
  const int halfSize = 20;
  for(size_t i=0;i<res.size();i++) {
    auto center = cetners[i];
    Rect rect;
    rect.bMin.x = center.y - halfSize;
    rect.bMin.y = center.x - halfSize;
    rect.bMax.x = center.y + halfSize;
    rect.bMax.y = center.x + halfSize;
    res[i] = rect;
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

std::vector<float3> AveragedColor4f(const float* data, int width, int heaight, const std::vector<Rect>& a_rectData)
{
  std::vector<float3> res(a_rectData.size());
  for(size_t rectId = 0; rectId < a_rectData.size(); rectId++) 
  { 
    auto rect = a_rectData[rectId];
    float3 summ(0,0,0);
    int pixelNum = 0;
    for(int y=rect.bMin.y; y<rect.bMax.y;y++) {
      for(int x = rect.bMin.x; x<rect.bMax.x;x++) {
        float pixelR = data[4*(y*width + x) + 0];
        float pixelG = data[4*(y*width + x) + 1];
        float pixelB = data[4*(y*width + x) + 2];
        summ.x += float(pixelR);
        summ.y += float(pixelG);
        summ.z += float(pixelB);
        pixelNum++;
      }
    }
    summ /= float(pixelNum);
    res[rectId] = summ; // / 255
  }

  return res;
}

std::vector<float> AveragedSpectrumFromImage3D(const float* data, int width, int height, int channels, const std::vector<Rect>& a_rectData)
{
  std::vector<float> allSpecters(channels*a_rectData.size());

  for(int c=0;c<channels;c++)
  {
    const float* imData = data + width*height*c;
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