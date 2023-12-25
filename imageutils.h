#pragma once

std::vector<float> LoadImage3d1f(const char* path, int* w, int* h, int* z);
bool SaveImage4fToEXR(const float* rgb, int width, int height, const char* outfilename, float a_normConst = 1.0f, bool a_invertY = false);
std::vector<float> LoadImage4fFromEXR(const char* infilename, int* pW, int* pH);
