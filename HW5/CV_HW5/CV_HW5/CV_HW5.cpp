#include "stdafx.h"
#include <iostream>
#include "CImg.h"
#include "iteration.h"
#include "otsu.h"

using namespace std;
using namespace cimg_library;

int main() {
	string path_head = "./pit/";
	string path_tail = ".jpg";
	for (int i = 1; i <= 107; i++) {
		string path = path_head + to_string(i) + path_tail;
		CImg<unsigned char> image(path.c_str());
		//iteration temp(image, i);
		//temp.seg();
		otsu temp(image, i);
		temp.seg();
	}
	system("pause");
    return 0;
}

