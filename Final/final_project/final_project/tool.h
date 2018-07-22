#include "CImg.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include "Eigen"

#define A4width 210
#define A4height 297

using namespace std;
using namespace cimg_library;
using namespace Eigen;

void gauss_filter(CImg<float>& filter, float sigma = 1.0f, int deriv = 0) {
	float width = 3 * sigma; 
	float sigma2 = sigma*sigma;
	filter.assign(int(2 * width) + 1);

	int i = 0;
	for (float x = -width; x <= width; x += 1.0f) {
		float g = exp(-0.5*x*x / sigma2) / sqrt(2 * cimg::PI) / sigma;
		if (deriv == 1) g *= -x / sigma2;
		if (deriv == 2) g *= (x*x / sigma2 - 1.0f) / sigma2;
		filter[i] = g;
		i++;
	}
}

CImg<unsigned char> preprocess(CImg<unsigned char>& img, int num) {
	CImg<unsigned char> result(img);
	cimg_forXY(img, x, y) {
		if (img(x, y) > num ) {
			result(x, y) = 0;
		}
		else {
			result(x, y) = 255;
		}
	}
	return result;
}

void CannyDiscrete(CImg<float> in, float sigma, float threshold,
	CImg<float> &outSmooth, CImg<float> &outGradient,
	CImg<float> &outOrientation, CImg<float> &outThreshold,
	CImg<float> &outNMS) {
	const int nx = in._width;
	const int ny = in._height;

	outGradient = in; outGradient.fill(0.0f);
	CImg<int> dirmax(outGradient);
	CImg<float> derivative[4];
	for (int i = 0; i < 4; i++) { derivative[i] = outGradient; }
	outOrientation = outGradient;
	outThreshold = outGradient;
	outNMS = outGradient;

	CImg<float> filter;
	gauss_filter(filter, sigma, 0);
	outSmooth = in.get_convolve(filter).convolve(filter.get_transpose());


	float fct = 1.0 / (2.0*sqrt(2.0f));
	for (int y = 1; y < ny - 1; y++) {
		for (int x = 1; x < nx - 1; x++) {
			float grad_E = (outSmooth(x + 1, y) - outSmooth(x - 1, y))*0.5; // E
			float grad_NE = (outSmooth(x + 1, y - 1) - outSmooth(x - 1, y + 1))*fct; // NE
			float grad_N = (outSmooth(x, y - 1) - outSmooth(x, y + 1))*0.5; // N
			float grad_SE = (outSmooth(x + 1, y + 1) - outSmooth(x - 1, y - 1))*fct; // SE

float grad_mag = grad_E*grad_E + grad_N*grad_N;
outGradient(x, y) = grad_mag;

float angle = 0.0f;
if (grad_mag > 0.0f) { angle = atan2(grad_N, grad_E); }
if (angle < 0.0) angle += cimg::PI;
outOrientation(x, y) = angle*255.0 / cimg::PI + 0.5;

derivative[0](x, y) = grad_E = fabs(grad_E);
derivative[1](x, y) = grad_NE = fabs(grad_NE);
derivative[2](x, y) = grad_N = fabs(grad_N);
derivative[3](x, y) = grad_SE = fabs(grad_SE);


if ((grad_E > grad_NE) && (grad_E > grad_N) && (grad_E > grad_SE)) {
	dirmax(x, y) = 0; // E
}
else if ((grad_NE > grad_N) && (grad_NE > grad_SE)) {
	dirmax(x, y) = 1; // NE
}
else if (grad_N > grad_SE) {
	dirmax(x, y) = 2; // N
}
else {
	dirmax(x, y) = 3; // SE
}
		}
	}

	int dir_vector[4][2] = { { 1,0 },{ 1,-1 },{ 0,-1 },{ 1,1 } };
	int dir, dir1, dir2;

	for (int y = 2; y < ny - 2; y++) {
		for (int x = 2; x < nx - 2; x++) {
			dir = dirmax(x, y);
			if (derivative[dir](x, y) < threshold) {
				outThreshold(x, y) = 0.0f;
				outNMS(x, y) = 0.0f;
			}
			else {
				outThreshold(x, y) = 255.0f;
				int dx = dir_vector[dir][0];
				int dy = dir_vector[dir][1];
				dir1 = dirmax(x + dx, y + dy);
				dir2 = dirmax(x - dx, y - dy);
				outNMS(x, y) = 255.f*
					((derivative[dir](x, y) > derivative[dir1](x + dx, y + dy)) &&
					(derivative[dir](x, y) >= derivative[dir2](x - dx, y - dy)));
			}
		}
	}
}

void hough(const CImg<float>& img, CImg<float>& Houghspace, CImg<unsigned char>& result, float in_thresh, int out_thresh, vector<pair<int, int>> &node) {
	const int width = img._width;
	const int height = img._height;

	int centerX = width / 2;
	int centerY = height / 2;

	int max_length = centerX * centerX + centerY * centerY;
	max_length = (int)sqrt(max_length + 0.0); //∫·÷·size

	const int  hough_space = 500;  //◊›÷·
	double hough_intervals = cimg::PI / (double)hough_space;

	Houghspace.assign(2 * max_length, hough_space);
	Houghspace.fill(0.0f);

	int** hough = new int*[500];
	for (int i = 0; i < hough_space; i++) {
		hough[i] = new int[2 * max_length]();
	}

	int max_hough = 0;
	cimg_forXY(img, x, y) {
		if (img(x, y) == 0)
			continue;
		for (int degree = 0; degree < hough_space; degree++) {
			double r = (x - centerX) * cos(degree * hough_intervals) + (y - centerY) * sin(degree * hough_intervals);
			r += max_length;
			if (r < 0 || r >= 2 * max_length)
				continue;
			Houghspace((unsigned int)r, degree)++;
			hough[degree][(int)r] ++;
			if (max_hough < hough[degree][(int)r])
				max_hough = hough[degree][(int)r];
		}
	}
	//Houghspace.display();
	//cout << max_hough << endl;

	int count = 0;
	vector<pair<int, int>> lines;
	cimg_forXY(Houghspace, x, y) {
		bool newlines = true;
		int temp = hough[y][x];
		if (temp > 183) {
			for (int k = 0; k < lines.size(); k++) {
				if ((abs(lines[k].first - y) < 15 && abs(lines[k].second - x) < 300) ||
					(abs(500 - lines[k].first - y) < 5 && abs(lines[k].second - x) < 480 && abs(lines[k].second - x) > 200)) {
					if (hough[y][x] > hough[lines[k].first][lines[k].second]) {
						lines[k].first = y;
						lines[k].second = x;
					}
					newlines = false;
				}
			}
			if (newlines) {
				lines.push_back(make_pair(y, x));
			}
		}
	}
	cout << lines.size() << endl;

	for (int i = 0; i < lines.size(); i++) {
		cout << lines[i].first << " " << lines[i].second << " " << hough[lines[i].first][lines[i].second] << endl;
	}

	for (int i = 0; i < lines.size(); i++) {
		int th = lines[i].first;
		int le = lines[i].second;
		if (th == 0) {
			cout << "line " << i << " : x = " << le - max_length << endl;
		}
		else if (th == 250) {
			cout << "line " << i << " : y = " << le - max_length << endl;
		}
		else {
			cout << "line " << i << " : y = " << -cos(th * hough_intervals) / sin(th * hough_intervals) << " * x + (";
			cout << (le - max_length) / sin(th * hough_intervals) << ")" << endl;
		}
	}

	unsigned char red[] = { 255, 0, 0 };
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i + 1; j < lines.size(); j++) {
			if (lines[i].first != 0 && lines[j].first != 0) {
				double a = -cos(lines[i].first * hough_intervals) / sin(lines[i].first * hough_intervals);
				double c = (lines[i].second - max_length) / sin(lines[i].first * hough_intervals);

				double b = -cos(lines[j].first * hough_intervals) / sin(lines[j].first * hough_intervals);
				double d = (lines[j].second - max_length) / sin(lines[j].first * hough_intervals);

				int real_x = (int)((d - c) / (a - b) + centerX);
				int real_y = (int)((a * d - b * c) / (a - b) + centerY);
				if (real_x > 0 && real_x < width && real_y > 0 && real_y < height) {
					result.draw_circle(real_x, real_y, 9, red, 1);
					node.push_back(make_pair(real_x, real_y));
				}
			}
			else if (lines[i].first == 0) {
				double b = -cos(lines[j].first * hough_intervals) / sin(lines[j].first * hough_intervals);
				double d = (lines[j].second - max_length) / sin(lines[j].first * hough_intervals);
				int real_x = lines[i].second - max_length + centerX;
				int real_y = (int)(b * real_x + d + centerY);
				if (real_x > 0 && real_x < width && real_y > 0 && real_y < height) {
					result.draw_circle(real_x, real_y, 9, red, 1);
					node.push_back(make_pair(real_x, real_y));
				}
			}
			else {
				int real_x = lines[j].first - max_length + centerX;
				double a = -cos(lines[i].first * hough_intervals) / sin(lines[i].first * hough_intervals);
				double c = (lines[i].second - max_length) / sin(lines[i].first * hough_intervals);
				int real_y = (int)(a * real_x + c + centerY);
				if (real_x > 0 && real_x < width && real_y > 0 && real_y < height) {
					result.draw_circle(real_x, real_y, 9, red, 1);
					node.push_back(make_pair(real_x, real_y));
				}
			}
		}
	}

	//ª≠œﬂ
	for (int i = 0; i < lines.size(); i++) {
		int y = lines[i].first;
		int x = lines[i].second;
		double dy = sin(y * hough_intervals);
		double dx = cos(y * hough_intervals);

		if ((y <= hough_space / 4) || (y >= 3 * hough_space / 4)) {
			for (int xtemp = 0; xtemp < height; xtemp++) {
				int ytemp;
				if (y == 0 || y == 500)
					ytemp = (int)(x - max_length) + centerX;
				else
					ytemp = (int)((x - max_length - ((xtemp - centerY) * dy)) / dx) + centerX;
				if (ytemp < width && ytemp >= 0) {
					if (result.atXY(ytemp, xtemp) == 255) {}
					//node.push_back(make_pair(ytemp, xtemp));
					else {
						result.atXY(ytemp, xtemp, 1) = 255;
						result.atXY(ytemp, xtemp, 0) = 0;
						result.atXY(ytemp, xtemp, 2) = 0;
					}
				}
			}
		}
		else {
			for (int sCol = 0; sCol < width; ++sCol) {
				int sRow;
				if (y == 250)sRow = (int)(x - max_length) + centerY;
				sRow = (int)((x - max_length - ((sCol - centerX) * dx)) / dy) + centerY;
				if (sRow < height && sRow >= 0) {
					if (result.atXY(sCol, sRow) == 255) {}
					//node.push_back(make_pair(sCol, sRow));
					else {
						result.atXY(sCol, sRow, 1) = 255;
						result.atXY(sCol, sRow, 0) = 0;
						result.atXY(sCol, sRow, 2) = 0;
					}
				}
			}
		}
	}

	//result.display();
	//cout << node.size() << endl;

}

CImg<unsigned char> paperChange(CImg<unsigned char> img, vector<pair<int, int>> node) {

	vector<pair<int, int>> order;
	int minDist, pointA;
	for (int i = 0; i < 4; i++) {
		int dist = node[i].first * node[i].first + node[i].second * node[i].second;
		if (i == 0) {
			minDist = dist;
			pointA = i;
		} else {
			if (dist < minDist) {
				minDist = dist;
				pointA = i;
			}
		}
	}

	int pointC = pointA;
	for (int i = 0; i < 4; i++) {
		if (i == pointC)
			continue;
		double k = (double)(node[i].second - node[pointA].second) / (double)(node[i].first - node[pointA].first);
		double b = (double)node[pointA].second - (double)node[pointA].first * k;

		int flag = 0;

		for (int j = 0; j < 4; j++) {
			if (j != i && j != pointA) {
				double diff = (double)node[j].second - (k * (double)node[j].first + b);
				if (flag == 0) {
					flag = diff > 0 ? 1 : -1;
				} else {
					if (flag == 1 && diff < 0 || flag == -1 && diff > 0) {
						pointC = i;
						break;
					}
				}
			}
		}
		if (pointC != pointA)
			break;
	}

	//cout << node[pointA].first << " " << node[pointA].second << endl;
	//cout << node[pointC].first << " " << node[pointC].second << endl;

	int type = 0;//1:∫·£¨2:–±£¨3: ˙÷±

	double k_ac = (double)(node[pointA].second - node[pointC].second) / (double)(node[pointA].first - node[pointC].first);
	if (abs(k_ac) < 1) {
		type = 1;
	} else {
		int x3 = -1, y3 = -1, x4 = -1, y4 = -1;
		for (int i = 0; i < 4; i++) {
			if (i != pointA && i != pointC) {
				if (x3 == -1) {
					x3 = node[i].first;
					y3 = node[i].second;
				} else {
					x4 = node[i].first;
					y4 = node[i].second;
				}
			}
		}
		double k_bd = (double)(y3 - y4) / (double)(x3 - x4);
		if (abs(k_bd) < 1) {
			type = 2;
		}
		else {
			type = 3;
		}
	}

	int pointB;
	int point3 = -1, point4 = -1;
	for (int i = 0; i < 4; i++) {
		if (i != pointA && i != pointC) {
			if (point3 == -1) {
				point3 = i;
			}
			else {
				point4 = i;
			}
		}
	}
	double dist3 = pow((node[point3].first - node[pointA].first), 2) + pow((node[point3].second - node[pointA].second), 2);
	double dist4 = pow((node[point4].first - node[pointA].first), 2) + pow((node[point4].second - node[pointA].second), 2);
	if (type == 3) {
		pointB = dist3 < dist4 ? point3 : point4;
	}
	else {
		pointB = dist3 > dist4 ? point3 : point4;
	}

	int pointD;
	for (int i = 0; i < 4; i++) {
		if (i != pointA && i != pointB && i != pointC) {
			pointD = i;
			break;
		}
	}
	order.push_back(node[pointA]);
	order.push_back(node[pointB]);
	order.push_back(node[pointC]);
	order.push_back(node[pointD]);

	for (int i = 0; i < 4; i++) {
		cout << "point" << i + 1 << " : " << order[i].first << " " << order[i].second << endl;
	}

	double kuan = sqrt(pow((order[0].first - order[1].first), 2) + pow((order[0].second - order[1].second), 2));
	double rate = kuan / A4width;
	double A4_w = A4width * rate;
	double A4_h = A4height * rate;
	CImg<unsigned char> result = CImg<unsigned char>((int)A4_w, (int)A4_h, 1, 3, 0);
	double V[8] = { 0,0,A4_w - 1,0,A4_w - 1,A4_h - 1,0,A4_h - 1 };

	MatrixXf A(8, 8);
	A << V[0], V[1], 1, 0, 0, 0, -V[0] * order[0].first, -V[1] * order[0].first,
		0, 0, 0, V[0], V[1], 1, -V[0] * order[0].second, -V[1] * order[0].second,
		V[2], V[3], 1, 0, 0, 0, -V[2] * order[1].first, -V[3] * order[1].first,
		0, 0, 0, V[2], V[3], 1, -V[2] * order[1].second, -V[3] * order[1].second,
		V[4], V[5], 1, 0, 0, 0, -V[4] * order[2].first, -V[5] * order[2].first,
		0, 0, 0, V[4], V[5], 1, -V[4] * order[2].second, -V[5] * order[2].second,
		V[6], V[7], 1, 0, 0, 0, -V[6] * order[3].first, -V[7] * order[3].first,
		0, 0, 0, V[6], V[7], 1, -V[6] * order[3].second, -V[7] * order[3].second;
	

	MatrixXf UV(8, 1);
	UV << order[0].first, order[0].second,
		order[1].first, order[1].second,
		order[2].first, order[2].second,
		order[3].first, order[3].second;
	MatrixXf H_(8, 1);
	H_ = A.inverse()*UV;

	cimg_forXY(result, x, y) {
		double u = (H_(0)*x + H_(1)*y + H_(2)) / (H_(6)*x + H_(7)*y + 1);
		double v = (H_(3)*x + H_(4)*y + H_(5)) / (H_(6)*x + H_(7)*y + 1);
		result(x, y, 0) = img(u, v, 0);
		result(x, y, 1) = img(u, v, 1);
		result(x, y, 2) = img(u, v, 2);
	}

	return result;
}

void houghline(CImg<unsigned char> img, vector<pair<int, int>> &node) {
	img.display();

	CImg<unsigned char> gray_img = img.get_RGBtoGray();
	float sigma = 1.5f;
	float threshold = 9;

	CImg<float> outS, outG, outO, outT, outNMS;
	CannyDiscrete(gray_img, sigma, threshold, outS, outG, outO, outT, outNMS);
	CImg<float> houghspace, output;
	float in_thresh = 200.0f;
	float out_thresh = 0.5f;
	hough(outNMS, houghspace, img, in_thresh, out_thresh, node);
}

void crop_line(CImg<unsigned char> img, vector<CImg<unsigned char>>& line) {
	int w = img._width;
	CImg<unsigned char> re = preprocess(img, 140);
	CImg<unsigned char> pit(w, img._height, 1, 1, 255);
	re.display();
	int index[18], k = 0;
	int num = 0, last_num = 0;
	bool cal = false;
	int total = 0;
	int first_temp;
	for (int i = 0; i < re._height; i++) {
		last_num = num;
		num = 0;
		for (int j = 0; j < re._width; j++) {
			if (re(j, i) == 255)
				num++;
		}
		for (int j = 0; j < num; j++) {
			pit(j, i) = 0;
		}
		if (last_num == 0 && num != 0) {
			cal = true;
			first_temp = i - 3;
		}
		if (cal) {
			total += num;
		}
		if (last_num != 0 && num == 0) {
			cal = false;
			if (total > 200) {
				index[k++] = first_temp;
				index[k++] = i + 3;
			}
			total = 0;
		}
		/*if (last_num == 0 && num > 10)
		index[k++] = i - 3;
		if (last_num != 0 && num >10)
		index[k++] = i + 3;*/
	}

	pit.display();

	CImg<unsigned char> region[9];
	for (int i = 0; i <= 8; i++) {
		cout << index[2 * i] << " " << index[2 * i + 1] << endl;
		region[i] = CImg<unsigned char>(w, index[2 * i + 1] - index[2 * i] + 1, 1, 1, 0);
		cimg_forXY(region[i], x, y) {
			region[i](x, y) = img(x, y + index[2 * i]);
		}
		//region[i].display();
		line.push_back(region[i]);
	}
}

void findmintag(int x, int y, int& mintag, CImg<int> tagimg) {
	if (tagimg(x, y - 1) != -1 && mintag > tagimg(x, y - 1)) {
		mintag = tagimg(x, y - 1);
	}

	if (tagimg(x - 1, y - 1) != -1 && mintag > tagimg(x - 1, y - 1)) {
		mintag = tagimg(x - 1, y - 1);
	}

	if (tagimg(x - 1, y) != -1 && mintag > tagimg(x - 1, y)) {
		mintag = tagimg(x - 1, y);
	}
	if (tagimg(x - 1, y + 1) != -1 && mintag > tagimg(x - 1, y + 1)) {
		mintag = tagimg(x - 1, y + 1);
	}
}

void merge(int x, int y, int mintag, CImg<int>& tagimg, vector<list<pair<int, int>>>& tagposlist) {
	if (y < 0 || x < 0 || x >= tagimg._width || y >= tagimg._height)
		return;
	if (tagimg(x, y) != -1) {
		//vector<vector<pair<int, int>>>::iterator iter = tagposlist.begin() + tagimg(x, y);
		int tag = tagimg(x, y);
		if (mintag != tag) {
			list<pair<int, int>>::iterator iter = tagposlist[tag].begin();
			for (; iter != tagposlist[tag].end(); iter++) {
				tagimg((*iter).first, (*iter).second) = mintag;
			}
			tagposlist[mintag].splice(tagposlist[mintag].end(), tagposlist[tag]);
		}
	}
}

void get_single_piture(CImg<unsigned char> img, CImg<int>& tagimg, vector<list<pair<int, int>>>& tagposlist) {
	cimg_forX(img, x) {
		cimg_forY(img, y) {
			if (x == 0 || y == 0) {
				if (img(x, y) == 255) {
					tagimg(x, y) = tagposlist.size();
					list<pair<int, int>> tagpos;
					tagpos.push_back(make_pair(x, y));
					tagposlist.push_back(tagpos);
				}
			}
			else {
				if (img(x, y) == 255) {
					int mintag = 10000;
					findmintag(x, y, mintag, tagimg);
					if (mintag == 10000) {
						tagimg(x, y) = tagposlist.size();
						list<pair<int, int>> tagpos;
						tagpos.push_back(make_pair(x, y));
						tagposlist.push_back(tagpos);
					}
					else {
						tagimg(x, y) = mintag;
						tagposlist[mintag].push_back(make_pair(x, y));
						merge(x, y - 1, mintag, tagimg, tagposlist);
						merge(x - 1, y - 1, mintag, tagimg, tagposlist);
						merge(x - 1, y, mintag, tagimg, tagposlist);
						merge(x - 1, y + 1, mintag, tagimg, tagposlist);
					}
				}
			}
		}
	}
}

void findBoundingbox(list<pair<int, int>> numlist, int& minx, int& miny, int& maxx, int& maxy) {
	minx = miny = 10000;
	maxx = maxy = 0;

	list<pair<int, int>>::iterator iter = numlist.begin();

	for (; iter != numlist.end(); iter++) {
		int x = (*iter).first;
		int y = (*iter).second;
		minx = x < minx ? x : minx;
		miny = y < miny ? y : miny;
		maxx = x > maxx ? x : maxx;
		maxy = y > maxy ? y : maxy;
	}
}

void crop_number(CImg<unsigned char> img, vector<list<pair<int, int>>> tagposlist, int index) {
	int totalpixel = 0, num = 0;
	for (int i = 0; i < tagposlist.size(); i++) {
		if (tagposlist[i].size() != 0) {
			totalpixel += tagposlist.size();
			num++;
		}
	}
	int avg = totalpixel / num;

	int j = 0;

	for (int i = 0; i < tagposlist.size(); i++) {
		if (tagposlist[i].size() != 0 && tagposlist[i].size() > 35) {
			int minx, miny, maxx, maxy;
			findBoundingbox(tagposlist[i], minx, miny, maxx, maxy);
			int w = maxx - minx;
			int h = maxy - miny;


			int size = w > h ? w : h;
			size += 10;
			CImg<unsigned char> number = CImg<unsigned char>(size, size, 1, 1, 0);

			list<pair<int, int>>::iterator iter = tagposlist[i].begin();
			for (; iter != tagposlist[i].end(); iter++) {
				int x = (*iter).first;
				int y = (*iter).second;
				int x_, y_;
				if (w > h) {
					x_ = x - minx + 5;
					y_ = y - miny + (size - h) / 2;
				}
				else {
					x_ = x - minx + (size - w) / 2;
					y_ = y - miny + 5;
				}
				number(x_, y_) = 255;
			}
			string path = "./img/" + to_string(index) + "_" + to_string(j) + ".bmp";
			j++;
			number = number.get_resize(28, 28);
			number.save(path.c_str());
			//number.display();
		}
	}
}