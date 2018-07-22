#include "stdafx.h"
#include <iostream>
#include "CImg.h"
#include <string>
#include "otsu.h"
#include <cmath>

using namespace std;
using namespace cimg_library;

otsu::otsu(CImg<unsigned char> in, int id_) {
	image = CImg<unsigned char>(in._width, in._height, 1, 3);
	result = CImg<unsigned char>(in._width, in._height, 1, 1);
	cimg_forXY(in, x, y) {
		image(x, y, 0) = in(x, y, 0);
		image(x, y, 1) = in(x, y, 1);
		image(x, y, 2) = in(x, y, 2);
	}
	id = id_;
	for (int i = 0; i < 256; i++) {
		color[i] = 0;
	}
	//image.display();
}

void otsu::getGary() {
	cimg_forXY(image, x, y) {
		result(x, y) = 0.299 * image(x, y, 0) + 0.587 * image(x, y, 1) + 0.114 * image(x, y, 2);
		int col = result(x, y);
		color[col] ++;
	}
	//result.display();
}

void otsu::findThreshold() {
	double max_g = 0;
	int best_t = 0;
	for (int i = 0; i <= 255; i++) {
		double w0 = 0;
		double w1 = 0;
		double u0 = 0;
		double u1 = 0;
		/*cimg_forXY(result, x, y) {
			if (result(x, y) < i) {
				w1 ++;
				u1 += result(x, y);
			} else {
				w0 ++;
				u0 += result(x, y);
			}
		}*/
		for (int j = 0; j < 256; j++) {
			if (j < i) {
				w1 += color[j];
				u1 += color[j] * j;
			} else {
				w0 += color[j];
				u0 += color[j] * j;
			}
		}
		int total = result._width * result._height;
		if (w1 != 0)
			u1 = u1 / w1;
		if (w0 != 0)
			u0 = u0 / w0;
		w1 = w1 / total;
		w0 = w0 / total;
		double g = w0 * w1 * pow((u0 - u1), 2);
		if (g > max_g) {
			max_g = g;
			best_t = i;
		}
	}
	threshold = best_t;
	//cout << best_t << endl;
	//cout << max_g << endl;
}

void otsu::seg() {
	getGary();
	findThreshold();
	cimg_forXY(result, x, y) {
		if (result(x, y) > threshold) {
			result(x, y) = 255;
		}
		else {
			result(x, y) = 0;
		}
	}
	string path = "./otsu_result1/";
	path += to_string(id);
	path += ".jpg";
	result.save(path.c_str());
	//cout << path << endl;
	//result.display();
}