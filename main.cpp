#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "checker.h"
#include "spectrum.h"
#include "imageutils.h"

using LiteMath::float4;

extern double __enzyme_autodiff(void*, ...);
int enzyme_const, enzyme_dup, enzyme_out;

double square(double x) {
    return x * x;
}
double dsquare(double x) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff((void*) square, x);
}

double loss(double* __restrict__ A, double* __restrict__ B, int n)
{
  double lossVal = 0.0;
  for(int i=0;i<n;i++) 
    lossVal += A[i]*A[i] + B[i]*B[i];
  return lossVal;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Функция с несколькими гауссианами
double f1(double x) {
    return exp(-(x - 2.0) * (x - 2.0) / 2.0) + 0.8 * exp(-(x - 5.0) * (x - 5.0) / 1.0) - 0.5 * exp(-(x - 8.0) * (x - 8.0) / 0.5);
}

// Функция с тремя гауссианами
double f2(double x) {
    return 0.7 * exp(-(x - 1.0) * (x - 1.0) / 1.0) + exp(-(x - 4.0) * (x - 4.0) / 0.5) - 0.4 * exp(-(x - 7.0) * (x - 7.0) / 2.0);
}

// Функция с четырьмя гауссианами
double f3(double x) {
    return 0.6 * exp(-(x - 1.5) * (x - 1.5) / 0.8) + exp(-(x - 4.0) * (x - 4.0) / 0.5) - 0.3 * exp(-(x - 6.5) * (x - 6.5) / 1.5) + 0.4 * exp(-(x - 9.0) * (x - 9.0) / 1.2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct TestData
{
  void Init(int a_size)
  {
    x_data.resize(a_size); 
    f1_data.resize(a_size);
    f2_data.resize(a_size);
    f3_data.resize(a_size);
    ref_data.resize(a_size);
    
    double x = 0.0;
    double step = 10/double(a_size);
    for(int i=0;i<a_size;i++,x+=step)
    {
      x_data  [i] = x;
      f1_data [i] = 1.0f;
      f2_data [i] = f2(x);
      f3_data [i] = f3(x);
      ref_data[i] = f1(x)*f2(x)*f3(x);
    }
  }

  std::vector<double> x_data; 
  std::vector<double> f1_data;
  std::vector<double> f2_data;
  std::vector<double> f3_data;
  std::vector<double> ref_data; // 
};

double TestDataLoss(double* __restrict__ f1_val, 
                    double* __restrict__ f2_val,
                    double* __restrict__ f3_val,
                    double* __restrict__ ref_val,
                    double* __restrict__ x_data, size_t n)
{
  double lossVal = 0.0;
  for(size_t i=0;i<n;i++) 
  { 
    double diff = f1_val[i]*f2_val[i]*f3_val[i] - ref_val[i];
    lossVal += diff*diff;
  }
  
  return lossVal; 
}

struct AdamOptimizer
{
  void Init(int a_size)
  {
    momentum.resize(a_size);
    grad.resize(a_size);
    m_GSquare.resize(a_size);
    std::fill(momentum.begin(), momentum.end(), 0.0);
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(m_GSquare.begin(), m_GSquare.end(), 0.0);
  }
  
  void UpdateState(double* a_state, int iter)
  {
    int factorGamma    = iter/100 + 1;
    const double alpha = 0.5;
    const double beta  = 0.25;
    const double gamma = 0.25/factorGamma;
    
    // Adam: m[i] = b*mPrev[i] + (1-b)*gradF[i], 
    // GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*grad[i]*grad[i])
    for(size_t i=0;i<grad.size();i++)
    {
      momentum [i] = momentum[i]*beta + grad[i]*(1.0-beta);
      m_GSquare[i] = 2.0*(m_GSquare[i]*alpha + (grad[i]*grad[i])*(1.0-alpha)); // does not works without 2.0
    }

    //xNext[i] = x[i] - gamma/(sqrt(GSquare[i] + epsilon)); 
    for (int i=0;i<grad.size();i++) 
      a_state[i] -= (gamma*momentum[i]/(std::sqrt(m_GSquare[i] + epsilon)));  
  }

  std::vector<double> momentum; 
  std::vector<double> grad;
  std::vector<double> m_GSquare;
  double epsilon = 1e-20f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*void testXYZ()
{
  auto rects    = GetCheckerRects();
  auto colorLDR = LoadAveragedCheckerLDRData("/home/frol/PROG/HydraRepos/HydraCore3/z_checker.bmp", rects); 
  
  int  channels = 0;
  auto colorHDR = LoadAveragedSpectrumFromImage3d1f("/home/frol/PROG/HydraRepos/HydraCore3/z_checker.image3d1f", rects, &channels); 

  auto spdLight = LoadAndResampleSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Lights/FalconEyesStudioLEDCOB120BW.spd",  channels); 
  auto spdMats  = LoadAndResampleAllCheckerSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/DatacolorSpyderCheckr24_card2", channels);
  
  Spectrum curveX, curveY, curveZ;
  {
    curveX.wavelengths = Get_CIE_lambda();
    curveX.values      = Get_CIE_X();
  
    curveY.wavelengths = Get_CIE_lambda();
    curveY.values      = Get_CIE_Y();
  
    curveZ.wavelengths = Get_CIE_lambda();
    curveZ.values      = Get_CIE_Z();
  }

  auto curveX1 = curveX.Resample(channels);
  auto curveY1 = curveY.Resample(channels);
  auto curveZ1 = curveZ.Resample(channels);

  int a = 2;
  std::cout << "curveX1.size() = " << curveX1.size() << std::endl;

  const float CIE_Y_integral = 106.856895f;

  // test spectrum image to RGB image
  //
  std::vector<float3> colorLDR2(rects.size());
  for(int rectId = 0; rectId < int(rects.size()); rectId++) {
    const float* colorSpec = colorHDR.data() + rectId*channels;
    float3 colorAccum(0,0,0);
    for(int c = 0; c < channels; c++) {
      float val = colorSpec[c];
      float lambda = LAMBDA_MIN + (float(c+0.5f)/float(channels))*(LAMBDA_MAX - LAMBDA_MIN);
      
      float3 xyz = SpectrumToXYZ(val, lambda, LAMBDA_MIN, LAMBDA_MAX,
                                 curveX.values.data(), curveY.values.data(), curveZ.values.data());
      
      //float3 xyz = {curveX1[c]*(val/pdf), curveY1[c]*(val/pdf), curveZ1[c]*(val/pdf)};
      float3 rgb = XYZToRGB(xyz);
      //if(rectId == 12) { 
      //  std::cout << "val,X = " << val << ", " << curveX1[c] << std::endl;
      //  //std::cout << "xyz = " << xyz.x << " " << xyz.y << " " << xyz.z << std::endl;
      //  //std::cout << "rgb = " << rgb.x << " " << rgb.y << " " << rgb.z << std::endl;
      //}
      colorAccum += rgb;
    }
    colorLDR2[rectId] = colorAccum*255.0f*(CIE_Y_integral/float(channels)); // * 100000.0f
  }

  for(size_t rectId = 0; rectId < colorLDR.size(); rectId++) {
    std::cout << "from_spd" << rectId << ":\t(" << colorLDR2[rectId].x << ", " << colorLDR2[rectId].y << ", " << colorLDR2[rectId].z << ")" << std::endl; 
  }
  
  std::cout << std::endl;
  for(size_t rectId = 0; rectId < colorLDR.size(); rectId++) {
    std::cout << "from_bmp" << rectId << ":\t(" << int(colorLDR[rectId].x+0.5f) << ", " << int(colorLDR[rectId].y+0.5f) << ", " << int(colorLDR[rectId].z+0.5f) << ")" << std::endl; 
    std::cout << "from_spd" << rectId << ":\t(" << int(colorLDR2[rectId].x+0.5f) << ", " << int(colorLDR2[rectId].y+0.5f) << ", " << int(colorLDR2[rectId].z+0.5f) << ")" << std::endl; 
    std::cout << std::endl;
  }
  std::cout << "channelNum = " << channels << std::endl;
}*/

void testGaussians()
{

  TestData data;
  AdamOptimizer opt;

  data.Init(100);  
  opt.Init(100);

  auto initial_f1 = data.f1_data;
  
  for(int iter = 0; iter < 50; iter++) 
  {
    std::fill(opt.grad.begin(), opt.grad.end(), 0.0);  

    double dloss = __enzyme_autodiff((void*)TestDataLoss,
                                     enzyme_dup,   data.f1_data.data(), opt.grad.data(),
                                     enzyme_const, data.f2_data.data(),
                                     enzyme_const, data.f3_data.data(),
                                     enzyme_const, data.ref_data.data(),
                                     enzyme_const, data.x_data.data(),
                                     enzyme_const, opt.grad.size());
    
    double lossVal = TestDataLoss(data.f1_data.data(),
                                  data.f2_data.data(),
                                  data.f3_data.data(),
                                  data.ref_data.data(),
                                  data.x_data.data(),
                                  opt.grad.size());                                     

    //opt.UpdateGrad();
    opt.UpdateState(data.f1_data.data(), iter);
    
    //if((iter+1) % 10 == 0)
    std::cout << "iter = " << iter << ", loss = (" << lossVal << ")" << std::endl;
  }

  std::ofstream fout2("data.csv");
  fout2 << "x;f1;f2;f3;f1xf2xf3;initial_f1;optimized_f1;" << std::endl;

  double x = 0.0;
  double step = 10/double(100);
  for(int i=0;i<100;i++,x+=step)
    fout2 << x << ";" << f1(x) << ";" << f2(x) << ";" << f3(x) << ";" << f1(x)*f2(x)*f3(x) << ";" << initial_f1[i] << ";" << data.f1_data[i] << ";" << std::endl;
  
}

void test3DImageToImage4f()
{
  int width = 0, height = 0, channels = 0;
  std::vector<float> image3d = LoadImage3d1f("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/LED_panel/8496/Images/render1.image3d1f", &width, &height, &channels);
  std::vector<float4> image2d(width*height);

  auto r = LoadAndResampleSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_r.spd",  channels);
  auto g = LoadAndResampleSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_g.spd",  channels);
  auto b = LoadAndResampleSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_b.spd",  channels); 

  std::cout << "width    = " << width << std::endl;
  std::cout << "height   = " << height << std::endl;
  std::cout << "channels = " << channels << std::endl;

  for(int y=0;y<height;y++) {
    for(int x=0;x<width;x++) {
      float4 color(0,0,0,0);
      for(int c=0;c<channels;c++) {
        int pixelAddr = y*width+x + c*width*height;
        float sVal = image3d[pixelAddr];
        color.x += r[c]*sVal;
        color.y += g[c]*sVal;
        color.z += b[c]*sVal;
      }
      color /= float(channels);  
      image2d[y*width+x] = color;    
    }
  }
  
  SaveImage4fToEXR((const float*)image2d.data(), width, height, "/home/frol/PROG/HydraRepos/rendervsphoto/Tests/LED_panel/8496/Images/render1_from_spd.exr");
}

float EvalRenderCoeff(const float* renderCoeff, const float* lgtSpec, const float* matSpec, const float* renderRef, int rectNum, int channelNum)
{
  float loss = 0.0f;
  for(int rectId = 0; rectId < rectNum; rectId++) {
    for(int c = 0; c < channelNum; c++) {
      float rend = renderRef[rectId*channelNum + c];
      float val  = rend - renderCoeff[rectId]*lgtSpec[c]*matSpec[rectId*channelNum + c];
      loss += val*val; 
    }
  }
  return loss;
}

float EvalRenderCoeff2(const float renderCoeff, const float* lgtSpec, const float* matSpec, const float* renderRef, int rectNum, int channelNum)
{
  float loss = 0.0f;
  for(int rectId = 0; rectId < rectNum; rectId++) {
    for(int c = 0; c < channelNum; c++) {
      float rend = renderRef[rectId*channelNum + c];
      float val  = rend - renderCoeff*lgtSpec[c]*matSpec[rectId*channelNum + c];
      loss += val*val; 
    }
  }
  return loss;
}

struct FuncData
{
  const float* lgtSpec;
  const float* matSpec;
  const float* renderRef;
  int rectNum;
  int channelNum;
};

float GoldenSectionCoeff2(float a, float b, float epsilon, FuncData data) 
{
  const float phi = (1 + sqrt(5)) / 2; // Золотое сечение
  float x1 = b - (b - a) / phi;
  float x2 = a + (b - a) / phi;
  while (std::abs(b - a) > epsilon) 
  {
    const float f1 = EvalRenderCoeff2(x1, data.lgtSpec, data.matSpec, data.renderRef, data.rectNum, data.channelNum);
    const float f2 = EvalRenderCoeff2(x2, data.lgtSpec, data.matSpec, data.renderRef, data.rectNum, data.channelNum);
    if (f1 < f2) {
        b = x2;
    } else {
        a = x1;
    }
    x1 = b - (b - a) / phi;
    x2 = a + (b - a) / phi;
  }
  return (a + b) / 2;
}

double EvalCurve1(double* camRGB, double* render, double* ref, size_t rectNum, int channelNum)
{
  float loss = 0.0f;
  for(size_t rectId = 0; rectId < rectNum; rectId++) {
    double valRef = ref[rectId];
    double rendVal = 0.0f;
    for(int c = 0; c < channelNum; c++) 
      rendVal += camRGB[c]*render[rectId*channelNum + c];
    rendVal /= double(channelNum);
    loss += (valRef - rendVal)*(valRef - rendVal); 
  }
  return loss;
}

/*
auto spdLight = LoadAndResampleSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Lights/FalconEyesStudioLEDCOB120BW.spd",  channels); 
auto spdMats  = LoadAndResampleAllCheckerSpectrum("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/DatacolorSpyderCheckr24_card2", channels);
std::vector<float> renderCoeff(rects.size(), 1.0f);
float loss = EvalRenderCoeff(renderCoeff.data(), spdLight.data(), spdMats.data(), avgSpec.data(), int(rects.size()), channels);
std::cout << "initial loss = " << loss << std::endl;

float minArgVal = GoldenSectionCoeff2(0.0f, 100.0f, 5e-6f, FuncData{spdLight.data(), spdMats.data(), avgSpec.data(), int(rects.size()), channels});
float minLoss   = EvalRenderCoeff2(minArgVal, spdLight.data(), spdMats.data(), avgSpec.data(), int(rects.size()), channels);
std::cout << "minLoss   = " << minLoss << std::endl;
std::cout << "minArgVal = " << minArgVal << std::endl;
*/


void FitSingleCurvePerImage(std::vector<Rect> rects,
                            const char* refImagePath,
                            const char* initialSpdPath, 
                            const char* image3dPath, 
                            const char* outPath, int rgbIndex, int leftBoundId = 2, int rightBoundId = 37)
{
  int width = 0, height = 0, channels = 0;
  std::vector<float> image3d = LoadImage3d1f(image3dPath, &width, &height, &channels);
  
  auto avgSpec    = AveragedSpectrumFromImage3D(image3d.data(), width, height, channels, rects);
  auto initialSpd = LoadAndResampleSpectrum(initialSpdPath,  channels);
  
  int w2, h2;
  auto image2dRef = LoadImage4fFromEXR(refImagePath, &w2, &h2);

  std::cout << "w2 = " << w2 << std::endl;
  std::cout << "h2 = " << h2 << std::endl;

  auto colors = AveragedColor4f(image2dRef.data(), w2, h2, rects);
  
  std::vector<double> curve(channels);
  std::vector<double> avgSpecD(avgSpec.size());
  std::vector<double> colorD(rects.size());
  {
    for(size_t i=0;i<curve.size();i++)
      curve[i] = double(initialSpd[i]);

    for(size_t i=0;i<avgSpecD.size();i++)
      avgSpecD[i] = double(avgSpec[i]);

    for(size_t i=0;i<colorD.size();i++)
      colorD[i] = double(colors[i][rgbIndex]);
  }

  double initialLossVal = EvalCurve1(curve.data(), avgSpecD.data(), colorD.data(), rects.size(), channels);
  std::cout << "initialLoss = " << initialLossVal << std::endl;
  
  
  AdamOptimizer opt;
  opt.Init(curve.size());
  
  for(int iter = 0; iter < 500; iter++) 
  {
    std::fill(opt.grad.begin(), opt.grad.end(), 0.0);  

    double dloss = __enzyme_autodiff((void*)EvalCurve1,
                                    enzyme_dup,   curve.data(), opt.grad.data(),
                                    enzyme_const, avgSpecD.data(),
                                    enzyme_const, colorD.data(),
                                    enzyme_const, rects.size(),
                                    enzyme_const, channels);
    
    double lossVal = EvalCurve1(curve.data(), avgSpecD.data(), colorD.data(), rects.size(), channels);                                   

    opt.UpdateState(curve.data(), iter);
    for(size_t i=0;i<curve.size();i++)
      if(curve[i] < 0.0)
        curve[i] = 0.0;
    
    std::cout << "iter = " << iter << ", loss = (" << lossVal << ")" << std::endl;
  }

  for(int i=0;i<=leftBoundId;i++)
    curve[i] = double(initialSpd[i]);
  for(int i=rightBoundId;i<int(curve.size());i++)
    curve[i] = double(initialSpd[i]);
  
  std::string outCSVName = std::string(outPath) + ".csv";
  std::string outSPDName = std::string(outPath) + ".spd";

  std::ofstream fout2(outCSVName.c_str()), fout3(outSPDName.c_str());
  fout2 << "lambda;initial;optimized;" << std::endl;

  float step = (LAMBDA_MAX - LAMBDA_MIN)/float(channels);
  for(size_t i=0;i<curve.size();i++) {
    float lambda = LAMBDA_MIN + step*float(i) + 0.5f*step;
    fout2 << lambda << ";" << initialSpd[i] << ";" << curve[i] << ";" << std::endl;
    fout3 << lambda << " " << curve[i] << std::endl;
  }
  fout2.close();

}

void OptRGBCurvesForCamMulLight(const char* a_testDir, const char* a_testNumber)
{
  std::string refImagePath = std::string(a_testDir) + "/" + std::string(a_testNumber) + "/Images/" + std::string("IMG_") + std::string(a_testNumber) + "_rawpy.exr";
  std::string spdImagePath = std::string(a_testDir) + "/" + std::string(a_testNumber) + "/Images/" + std::string("render1.image3d1f");
  std::string mskImagePath = std::string(a_testDir) + "/" + std::string(a_testNumber) + "/" + std::string("mask.png");

  auto rects = GetCheckerRectFromMask(mskImagePath.c_str());

  FitSingleCurvePerImage(rects, refImagePath.c_str(),  "/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_r.spd", 
                         spdImagePath.c_str(), "Canon60D_r_opt", 0);

  FitSingleCurvePerImage(rects, refImagePath.c_str(),  "/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_g.spd", 
                         spdImagePath.c_str(), "Canon60D_g_opt", 1);
                  
  FitSingleCurvePerImage(rects, refImagePath.c_str(),  "/home/frol/PROG/HydraRepos/rendervsphoto/Tests/data/Spectral_data/Camera/Canon60D_b.spd", 
                         spdImagePath.c_str(), "Canon60D_b_opt", 2);
}


int main(int argc, const char** argv) 
{
  //test3DImageToImage4f();
  
  //OptRGBCurvesForCamMulLight("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/FalconEyesStudioLEDCOB120BW", "8459");

  OptRGBCurvesForCamMulLight("/home/frol/PROG/HydraRepos/rendervsphoto/Tests/LED_panel", "8496");


  return 0;
}
