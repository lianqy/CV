#include "stdafx.h"
#include <iostream>
#include "CImg.h"

using namespace std;
using namespace cimg_library;


int main()
{
	//读取图像
	CImg<unsigned char> SrcImg;
	SrcImg.load_bmp("1.bmp");
	int w = SrcImg._width;
	int h = SrcImg._height;

	//显示图像
	SrcImg.display();

	//将白色区域变成红色,黑色区域变绿色
	CImg<unsigned char> Tar_Img("1.bmp");

	cimg_forXY(SrcImg, x, y) {
		if (SrcImg(x, y, 0) == 255 && SrcImg(x, y, 1) == 255 && SrcImg(x, y, 2) == 255) {
			Tar_Img(x, y, 0) = 255;
			Tar_Img(x, y, 1) = 0;
			Tar_Img(x, y, 2) = 0;
		} else if (SrcImg(x, y, 0) == 0 && SrcImg(x, y, 1) == 0 && SrcImg(x, y, 2) == 0) {
			Tar_Img(x, y, 0) = 0;
			Tar_Img(x, y, 1) = 255;
			Tar_Img(x, y, 2) = 0;
		}
	}

	//绘制圆形，坐标为（50,50），半径为30，填充蓝色。半径为3，填充黄色。
	unsigned char blue[] = {0, 0, 255};
	unsigned char yellow[] = { 255, 255, 0 };
	Tar_Img.draw_circle(50, 50, 30, blue, 1);
	Tar_Img.draw_circle(50, 50, 3, yellow, 1);

	Tar_Img.blur_median(3);

	Tar_Img.display();
	Tar_Img.save("result.bmp");

    return 0;
}

