#include <iostream>
#include "CImg.h"
#include <string>

using namespace std;
using namespace cimg_library;

class otsu {
private:
	CImg<unsigned char> image;
	CImg<unsigned char> result;
	int threshold;
	int id;
	int color[256];
public:
	otsu(CImg<unsigned char> in, int id_);
	void getGary();
	void findThreshold();
	void seg();
};