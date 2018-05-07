#include <iostream>
#include "CImg.h"
#include "tool.h"

using namespace std;
using namespace cimg_library;

void houghline() {
	CImg<unsigned char> img;
	img.load_bmp("./Dataset1/2.bmp");
	img = img.get_resize(img._width * 0.25, img._height * 0.25);
	img.display();

	CImg<unsigned char> gray_img = preprocess(img, 100);
	//gray_img.display();

	gray_img = gray_img.get_RGBtoGray();
	//CImg<unsigned char> gray_img = img.get_RGBtoGray();
	//gray_img.display();

	//gray_img = gray_img.get_threshold(100);
	//preprocess(gray_img, 115);
	//gray_img.display();

	float sigma = 1.5f;
	float threshold = 9;

	CImg<float> outS, outG, outO, outT, outNMS;
	CannyDiscrete(gray_img, sigma, threshold, outS, outG, outO, outT, outNMS);

	//outS.display();
	//outG.display();
	//outO.display();
	//outT.display();
	outNMS.display();

	CImg<float> houghspace, output;
	float in_thresh = 200.0f;
	float out_thresh = 0.5f;
	hough(outNMS, houghspace, img, in_thresh, out_thresh);
	img.display();
}

void houghcircle() {
	CImg<unsigned char> img;
	img.load_bmp("./Dataset2/2.bmp");
	if (img._width > 999) {
		int rate = img._width / 500;
		double sca = 1 / (double)rate;
		img = img.get_resize((int)img._width * sca, (int)img._height * sca);
	}
	unsigned char red[] = { 255, 0, 0 };
	//img.draw_circle(320, 220, 150, red, 1, 0);
	img.display();

	CImg<unsigned char> gray_img = img.get_RGBtoGray();
	//gray_img.display();

	float sigma = 1.0f;
	float threshold = 10;

	CImg<float> outS, outG, outO, outT, outNMS;
	CannyDiscrete(gray_img, sigma, threshold, outS, outG, outO, outT, outNMS);

	//outS.display();
	//outG.display();
	//outO.display();
	//outT.display();
	outNMS.display();
	
	CImg<float> houghspace_;
	hough_circle(outNMS, houghspace_, img, 0, 0);
}

int main() {
	houghline();
	//houghcircle();
	system("pause");
	return 0;
}