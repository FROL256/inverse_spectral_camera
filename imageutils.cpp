#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#if defined(_WIN32)
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
#endif

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

bool SaveImage4fToEXR(const float* rgb, int width, int height, const char* outfilename, float a_normConst = 1.0f, bool a_invertY = false) 
{
  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);
  image.num_channels = 3;

  std::vector<float> images[3];
  images[0].resize(width * height);
  images[1].resize(width * height);
  images[2].resize(width * height);

  // Split RGBARGBARGBA... into R, G and B layer
  if(a_invertY) {
    for(int y=0;y<height;y++) {
      const int offsetY1 = y*width*4;
      const int offsetY2 = (height-y-1)*width*4;
      for(int x=0;x<width;x++) {
        images[0][(offsetY1 >> 2) + x] = rgb[offsetY2 + x*4 + 0]*a_normConst;
        images[1][(offsetY1 >> 2) + x] = rgb[offsetY2 + x*4 + 1]*a_normConst;
        images[2][(offsetY1 >> 2) + x] = rgb[offsetY2 + x*4 + 2]*a_normConst; 
      }
    }   
  }
  else {
    for (size_t i = 0; i < size_t(width * height); i++) {
      images[0][i] = rgb[4*i+0]*a_normConst;
      images[1][i] = rgb[4*i+1]*a_normConst;
      images[2][i] = rgb[4*i+2]*a_normConst;
    }
  }

  float* image_ptr[3];
  image_ptr[0] = images[2].data(); // B
  image_ptr[1] = images[1].data(); // G
  image_ptr[2] = images[0].data(); // R

  image.images = (unsigned char**)image_ptr;
  image.width  = width;
  image.height = height;
  header.num_channels = 3;
  header.channels     = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
  strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
  strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

  header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
  }
 
  const char* err = nullptr; 
  int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err); // free's buffer for an error message
    return false;
  }
  //printf("Saved exr file. [%s] \n", outfilename);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  return true;
}

std::vector<float> LoadImage3d1f(const char* path, int* w, int* h, int* z)
{
  std::ifstream fin(path, std::ios::binary);
  int xyz[3] = {};
  fin.read((char*)xyz, sizeof(int)*3);

  const int width    = xyz[0];
  const int height   = xyz[1];
  const int channels = xyz[2];

  if(w != nullptr) (*w) = width;
  if(h != nullptr) (*h) = height;
  if(z != nullptr) (*z) = channels;

  std::vector<float> data(width*height*channels);
  fin.read((char*)data.data(), sizeof(float)*data.size());
  fin.close();
  return data;
}

std::vector<float> LoadImage4fFromEXR(const char* infilename, int* pW, int* pH) 
{
  std::vector<float> result;
  float* out; // width * height * RGBA
  int width  = 0;
  int height = 0;
  const char* err = nullptr; 

  int ret = LoadEXR(&out, &width, &height, infilename, &err);
  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      fprintf(stderr, "[LoadImage4fFromEXR] : %s\n", err);
      std::cerr << "[LoadImage4fFromEXR] : " << err << std::endl;
      delete err;
    }
  }
  else {
    result.resize(width * height*4);
    *pW = uint32_t(width);
    *pH = uint32_t(height);
    memcpy(result.data(), out, width*height*sizeof(float)*4);
    free(out);
  }
  
  return result;
}