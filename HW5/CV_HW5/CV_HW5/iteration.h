#include <iostream>
#include "CImg.h"
#include <string>

using namespace std;
using namespace cimg_library;

class iteration {
private:
	int T;
	CImg<unsigned char> image;
	CImg<unsigned char> result;
	int threshold;
	int id;
public:
	iteration(CImg<unsigned char> in, int id_);
	void getGary();
	void findThreshold();
	void seg();
};