#include "CImg.h"
#include "stdafx.h"

using namespace cimg_library;

#ifndef canny_h
#define canny_h

CImg<unsigned char> canny(CImg<unsigned char> img);

CImg<unsigned char> cannyparam(CImg<unsigned char> img, float lowthreshold, float highthreshold,
							   float gaussiankernelradius, int gaussiankernelwidth,
							   int contrastnormalised);

CImg<unsigned char> Luminance(CImg<unsigned char> img);

#endif