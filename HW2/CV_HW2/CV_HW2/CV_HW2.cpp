#include "stdafx.h"
#include <iostream>
#include "Canny.h"

using namespace std;

int main() {	
	CImg<unsigned char> test1, test2, test3, test4;

	test1.load_bmp("./test_Data/lena.bmp");
	test1.display();
	test1 = canny(test1);
	test1.display();

	test2.load_bmp("./test_Data/bigben.bmp");
	test2.display();
	test2 = canny(test2);
	test2.display();

	test3.load_bmp("./test_Data/stpietro.bmp");
	test3.display();
	test3 = canny(test3);
	test3.display();

	test4.load_bmp("./test_Data/twows.bmp");
	test4.display();
	test4 = canny(test4);
	test4.display();

	system("pause");
    return 0;
}

