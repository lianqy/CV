#include "stdafx.h"
#include <iostream>
#include "CImg.h"

using namespace std;
using namespace cimg_library;


int main()
{
	//��ȡͼ��
	CImg<unsigned char> SrcImg;
	SrcImg.load_bmp("1.bmp");
	int w = SrcImg._width;
	int h = SrcImg._height;

	//��ʾͼ��
	SrcImg.display();

	//����ɫ�����ɺ�ɫ,��ɫ�������ɫ
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

	//����Բ�Σ�����Ϊ��50,50�����뾶Ϊ30�������ɫ���뾶Ϊ3������ɫ��
	unsigned char blue[] = {0, 0, 255};
	unsigned char yellow[] = { 255, 255, 0 };
	Tar_Img.draw_circle(50, 50, 30, blue, 1);
	Tar_Img.draw_circle(50, 50, 3, yellow, 1);

	Tar_Img.blur_median(3);

	Tar_Img.display();
	Tar_Img.save("result.bmp");

    return 0;
}

