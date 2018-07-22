#include "stdafx.h"
#include <iostream>
#include "CImg.h"
#include <string>
#include "iteration.h"
#include <cmath>

using namespace std;
using namespace cimg_library;

iteration::iteration(CImg<unsigned char> in, int id_) {
	image = CImg<unsigned char>(in._width, in._height, 1, 3);
	result = CImg<unsigned char>(in._width, in._height, 1, 1);
	cimg_forXY(in, x, y) {
		image(x, y, 0) = in(x, y, 0);
		image(x, y, 1) = in(x, y, 1);
		image(x, y, 2) = in(x, y, 2);
	}
	T = 127;
	threshold = 0.1;
	id = id_;
	//image.display();
}

void iteration::getGary() {
	cimg_forXY(image, x, y) {
		result(x, y) = 0.299 * image(x, y, 0) + 0.587 * image(x, y, 1) + 0.114 * image(x, y, 2);
	}
	//result.display();
}

void iteration::findThreshold() {
	int nT = 0;
	while (abs(nT - T) > threshold) {
		int G1 = 0, G2 = 0;
		int num1 = 0, num2 = 0;
		cimg_forXY(result, x, y) {
			if (result(x, y) >= T) {
				G1 += result(x, y);
				num1++;
			} else {
				G2 += result(x, y);
				num2++;
			}
		}
		int m1 = G1 / num1;
		int m2 = G2 / num2;
		nT = T;
		T = (m1 + m2) / 2;
	}
}

void iteration::seg() {
	getGary();
	findThreshold();
	cimg_forXY(result, x, y) {
		if (result(x, y) > T) {
			result(x, y) = 255;
		} else {
			result(x, y) = 0;
		}
	}
	string path = "./it_result/";
	path += to_string(id);
	path += ".jpg";
	result.save(path.c_str());
	//cout << path << endl;
	//result.display();
}