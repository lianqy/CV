#include <iostream>
#include "CImg.h"
#include "tool.h"
#include "Eigen"

using namespace std;
using namespace cimg_library;
using namespace Eigen;

int main() {
	CImg<unsigned char> img("./image/15331180.jpg");
	CImg<unsigned char> img_temp(img);
	if (img._width > 3000) {
		float scale = 2500.0 / img._width;
		img = img.get_resize(img._width * scale, img._height * scale);
	}
	if (img._width > img._height) {
		img = CImg<unsigned char>(img_temp._height, img_temp._width,1,3);
		cimg_forY(img, y) {
			cimg_forX(img, x) {
				img(x, y, 0) = img_temp(y, img_temp._height - 1 - x, 0);
				img(x, y, 1) = img_temp(y, img_temp._height - 1 - x, 1);
				img(x, y, 2) = img_temp(y, img_temp._height - 1 - x, 2);
			}
		}
	}
	vector<pair<int, int>> node;
	CImg<unsigned char> result;

	houghline(img, node);
	result = paperChange(img, node);
	result.display();
	result = result.get_crop(10, 10, result._width - 10, result._height - 10);
	result = result.get_RGBtoGray();
	result.display();

	vector<CImg<unsigned char>> line;
	crop_line(result, line);

	for (int i = 0; i < line.size(); i++) {
		line[i] = preprocess(line[i], 150);
		line[i] = line[i].dilate(2); //ÅòÕÍ
		line[i].display();

		vector<list<pair<int, int>>> tagposlist;

		CImg<int> tagimg = CImg<int>(line[i]._width, line[i]._height, 1, 1, -1);
		get_single_piture(line[i], tagimg, tagposlist);
		tagimg.display();
		crop_number(line[i], tagposlist, i);
	}

	system("pause");
	return 0;
}