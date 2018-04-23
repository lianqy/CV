#include "Canny.h"
#include "stdafx.h"
#include "CImg.h"
#include <cmath>
#include <vector>

using namespace cimg_library;
using namespace std;

#define ffabs(x) ( (x) >= 0 ? (x) : -(x) )
#define GAUSSIAN_CUT_OFF 0.005f
#define MAGNITUDE_SCALE 100.0f
#define MAGNITUDE_LIMIT 1000.0f
#define MAGNITUDE_MAX ((int) (MAGNITUDE_SCALE * MAGNITUDE_LIMIT))

class CANNY {
public:
	CImg<unsigned char> data;
	CImg<int> idata;
	CImg<int> magnitude;
	CImg<float> xConv;
	CImg<float> yConv;
	CImg<float> xGradient;
	CImg<float> yGradient;

	CANNY(int width, int height) {
		data = CImg<unsigned char>(width, height, 1, 1);
		idata = CImg<int>(width, height, 1, 1);
		magnitude = CImg<int>(width, height, 1, 1);
		xConv = CImg<float>(width, height, 1, 1);
		yConv = CImg<float>(width, height, 1, 1);
		xGradient = CImg<float>(width, height, 1, 1);
		yGradient = CImg<float>(width, height, 1, 1);
	}
};

static float hypotenuse(float x, float y);
static void performHysteresis(CANNY &can, int low, int high);
static  void follow(CANNY &can, int x1, int y1, int i1, int threshold);
CImg<unsigned char> canny(CImg<unsigned char> img);
CImg<unsigned char> cannyparam(CImg<unsigned char> img, float lowthreshold, float highthreshold,
	float gaussiankernelradius, int gaussiankernelwidth,
	int contrastnormalised);
CImg<unsigned char> Luminance(CImg<unsigned char> img);
static float gaussian(float x, float sigma);
int computeGradients(CANNY &can, float kernelRadius, int kernelWidth);
static void normalizeContrast(CANNY & can);

CImg<unsigned char> canny(CImg<unsigned char> img) {
	return cannyparam(img, 2.5f, 7.5f, 2.0f, 16, 0);
}

CImg<unsigned char> cannyparam(CImg<unsigned char> img, float lowthreshold, float highthreshold,
	float gaussiankernelradius, int gaussiankernelwidth,
	int contrastnormalised) {
	int low, high;
	int err;
	int i;

	int width = img._width;
	int height = img._height;
	CANNY can(width, height);
	CImg<unsigned char> answer(width, height, 1, 1);

	can.data = Luminance(img); //转换成灰度图

	if (contrastnormalised)
		normalizeContrast(can);

	err = computeGradients(can, gaussiankernelradius, gaussiankernelwidth); //计算梯度

	low = (int)(lowthreshold * MAGNITUDE_SCALE + 0.5f);
	high = (int)(highthreshold * MAGNITUDE_SCALE + 0.5f);
	performHysteresis(can, low, high);
	for (i = 0;i < width * height; i++)
		answer(i%width, i / width) = can.idata(i%width, i/width) > 0 ? 255 : 0;
	return answer;
}

static void normalizeContrast(CANNY & can) {
	int width = can.data._width;
	int height = can.data._height;
	int histogram[256] = { 0 };
	int remap[256];
	int sum = 0;
	int j = 0;
	int k;
	int target;
	int i;

	for (i = 0; i < width * height; i++)
		histogram[can.data(i%width, i/width)]++;


	for (i = 0; i < 256; i++) {
		sum += histogram[i];
		target = (sum * 255) / (width * height);
		for (k = j + 1; k <= target; k++)
			remap[k] = i;
		j = target;
	}

	for (i = 0; i < width * height; i++)
		can.data(i%width, i / width) = remap[can.data(i%width, i / width)];
}

int computeGradients(CANNY &can, float kernelRadius, int kernelWidth) {
	int width = (can.data)._width;
	int height = (can.data)._height;

	vector<float> kernel(kernelWidth, 0);
	vector<float> diffkernel(kernelWidth, 0);

	int kwidth;

	//初始化kernel
	for (kwidth = 0; kwidth < kernelWidth; kwidth++) {
		float g1, g2, g3;
		g1 = gaussian((float)kwidth, kernelRadius);
		if (g1 <= GAUSSIAN_CUT_OFF && kwidth >= 2)
			break;
		g2 = gaussian(kwidth - 0.5f, kernelRadius);
		g3 = gaussian(kwidth + 0.5f, kernelRadius);
		kernel[kwidth] = (g1 + g2 + g3) / 3.0f / (2.0f * (float) 3.14 * kernelRadius * kernelRadius);
		diffkernel[kwidth] = g3 - g2;
	}

	int initX = kwidth - 1;
	int maxX = width - (kwidth - 1);
	int initY = width * (kwidth - 1);
	int maxY = width * (height - (kwidth - 1));
	int x, y;

	for (x = initX; x < maxX; x++) {
		for (y = initY; y < maxY; y+= width) {
			int index = x + y;
			int index_x = index % width;
			int index_y = index / width;
			float sumX = (can.data(index_x, index_y)) * kernel[0];
			float sumY = sumX;
			int xOffset = 1;
			int yOffset = width;
			while (xOffset < kwidth) {
				sumY += kernel[xOffset] * (can.data((index - yOffset)%width, (index - yOffset) / width) + can.data((index + yOffset) % width, (index + yOffset) / width));
				sumX += kernel[xOffset] * (can.data((index - xOffset) % width, (index - xOffset) / width) + can.data((index + xOffset) % width, (index + xOffset) / width));
				yOffset += width;
				xOffset++;
			}

			can.yConv(index%width, index/width) = sumY;
			can.xConv(index % width, index/width) = sumX;
		}
	}
	
	for (x = initX; x < maxX; x++) {
		for (y = initY; y < maxY; y += width) {
			float sum = 0;
			int index = x + y;
			for (int i = 1; i < kwidth; i++) {
				sum += diffkernel[i] * (can.yConv((index - i)%width, (index - i)/width) - can.yConv((index + i) % width, (index + i) / width));
			}
			can.xGradient(index % width, index / width) = sum;
		}
	}

	for (x = kwidth; x < width - kwidth; x++) {
		for (y = initY; y < maxY; y += width) {
			float sum = 0;
			int index = x + y;
			int yOffset = width;
			for (int i = 1; i < kwidth; i++) {
				sum += diffkernel[i] * (can.xConv((index - yOffset) % width, (index - yOffset) / width) - can.xConv((index + yOffset) % width, (index + yOffset) / width));
				yOffset += width;
			}
			can.yGradient(index % width, index / width) = sum;
		}
	}

	initX = kwidth;
	maxX = width - kwidth;
	initY = width * kwidth;
	maxY = width * (height - kwidth);

	for (x = initX; x < maxX; x++) {
		for (y = initY; y < maxY; y += width) {
			int index = x + y;
			int indexN = index - width;
			int indexS = index + width;
			int indexW = index - 1;
			int indexE = index + 1;
			int indexNW = indexN - 1;
			int indexNE = indexN + 1;
			int indexSW = indexS - 1;
			int indexSE = indexS + 1;

			float xGrad = can.xGradient(index%width, index/width);
			float yGrad = can.yGradient(index%width, index/width);
			float gradMag = hypotenuse(xGrad, yGrad);

			float nMag = hypotenuse(can.xGradient(indexN % width, indexN/width), can.yGradient(indexN % width, indexN/width));
			float sMag = hypotenuse(can.xGradient(indexS % width, indexS/width), can.yGradient(indexS % width, indexS/width));
			float wMag = hypotenuse(can.xGradient(indexW % width, indexW/width), can.yGradient(indexW % width, indexW/width));
			float eMag = hypotenuse(can.xGradient(indexE % width, indexE/width), can.yGradient(indexE % width, indexE/width));
			float neMag = hypotenuse(can.xGradient(indexNE % width, indexNE/width), can.yGradient(indexNE % width, indexNE/width));
			float seMag = hypotenuse(can.xGradient(indexSE % width, indexSE/width), can.yGradient(indexSE % width, indexSE/width));
			float swMag = hypotenuse(can.xGradient(indexSW % width, indexSW/width), can.yGradient(indexSW % width, indexSW/width));
			float nwMag = hypotenuse(can.xGradient(indexNW % width, indexNW/width), can.yGradient(indexNW % width, indexNW/width));

			float tmp;

			int flag = ((xGrad * yGrad <= 0.0f)
				? ffabs(xGrad) >= ffabs(yGrad)
				? (tmp = ffabs(xGrad * gradMag)) >= ffabs(yGrad * neMag - (xGrad + yGrad) * eMag)
				&& tmp > ffabs(yGrad * swMag - (xGrad + yGrad) * wMag)
				: (tmp = ffabs(yGrad * gradMag)) >= ffabs(xGrad * neMag - (yGrad + xGrad) * nMag)
				&& tmp > ffabs(xGrad * swMag - (yGrad + xGrad) * sMag) 
				: ffabs(xGrad) >= ffabs(yGrad) 
				? (tmp = ffabs(xGrad * gradMag)) >= ffabs(yGrad * seMag + (xGrad - yGrad) * eMag) 
				&& tmp > ffabs(yGrad * nwMag + (xGrad - yGrad) * wMag) 
				: (tmp = ffabs(yGrad * gradMag)) >= ffabs(xGrad * seMag + (yGrad - xGrad) * sMag) 
				&& tmp > ffabs(xGrad * nwMag + (yGrad - xGrad) * nMag) 
				);

			if (flag) {
				can.magnitude(index % width, index/width) = (gradMag >= MAGNITUDE_LIMIT) ? MAGNITUDE_MAX : (int)(MAGNITUDE_SCALE * gradMag);
			} else {
				can.magnitude(index % width, index/width) = 0;
			}
		}
	}
	return 0;
}

static float hypotenuse(float x, float y) {
	return (float)sqrt(x*x + y*y);
}

static void performHysteresis(CANNY &can, int low, int high) {
	int offset = 0;
	int x, y;
	int width = (can.data)._width;
	int height = (can.data)._height;

	(can.idata).resize(width, height, 1, 1);
	(can.idata).fill(0);

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			if (can.idata(offset%width, offset/width) == 0 && can.magnitude(offset%width, offset/width) >= high)
				follow(can, x, y, offset, low);
			offset++;
		}
	}
}

static  void follow(CANNY &can, int x1, int y1, int i1, int threshold) {
	int x, y;
	int width = (can.data)._width;
	int height = (can.data)._height;
	int x0 = x1 == 0 ? x1 : x1 - 1;
	int x2 = x1 == width - 1 ? x1 : x1 + 1;
	int y0 = y1 == 0 ? y1 : y1 - 1;
	int y2 = y1 == height - 1 ? y1 : y1 + 1;

	can.idata(i1%width, i1/width) = can.magnitude(i1 % width, i1/width);
	for (x = x0; x <= x2; x++)
	{
		for (y = y0; y <= y2; y++)
		{
			int i2 = x + y * width;
			if ((y != y1 || x != x1) && can.idata(i2 % width, i2/width) == 0 && can.magnitude(i2 % width, i2/width) >= threshold)
				follow(can, x, y, i2, threshold);
		}
	}
}

CImg<unsigned char> Luminance(CImg<unsigned char> img) {
	int width = img._width;
	int height = img._height;
	CImg<unsigned char> gray(width, height, 1, 1);
	unsigned char gray_temp = 0;
	cimg_forXY(img, x, y) {
		gray_temp = 0.299 * img(x, y, 0) + 0.587 * img(x, y, 1) + 0.114 * img(x, y, 2);
		//gray(x, y, 0) = gray_temp;
		//gray(x, y, 1) = gray_temp;
		//gray(x, y, 2) = gray_temp;
		gray(x, y) = gray_temp;
	}
	return gray;
}

static float gaussian(float x, float sigma) {
	return (float)exp(-(x * x) / (2.0f * sigma * sigma));
}