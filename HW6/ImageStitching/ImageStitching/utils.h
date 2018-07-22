#pragma once

#include "CImg.h"
#include <vector>
#include "vl/sift.h"
#include "vl/generic.h"
#include "vl/kdtree.h"
#include <map>
#include <cmath>
#include <ctime>
#include <set>
#include "Eigen"

#define Max(a,b)(a>b?a:b)

using namespace std;
using namespace cimg_library;
using namespace Eigen;

struct HomographyMatrix {
	float a, b, c, d, e, f, g, h;
	HomographyMatrix(float _a, float _b, float _c,
		float _d, float _e, float _f, float _g, float _h) :
		a(_a), b(_b), c(_c), d(_d), e(_e), f(_f), g(_g), h(_h) {}
};

struct keypointpair {
	VlSiftKeypoint point1;
	VlSiftKeypoint point2;
	keypointpair(VlSiftKeypoint p1, VlSiftKeypoint p2) : point1(p1), point2(p2) {}
};

HomographyMatrix get_homography_matrix(const vector<keypointpair>& pair) {
	assert(pair.size() == 4);

	float u0 = pair[0].point1.x, v0 = pair[0].point1.y;
	float u1 = pair[1].point1.x, v1 = pair[1].point1.y;
	float u2 = pair[2].point1.x, v2 = pair[2].point1.y;
	float u3 = pair[3].point1.x, v3 = pair[3].point1.y;

	float x0 = pair[0].point2.x, y0 = pair[0].point2.y;
	float x1 = pair[1].point2.x, y1 = pair[1].point2.y;
	float x2 = pair[2].point2.x, y2 = pair[2].point2.y;
	float x3 = pair[3].point2.x, y3 = pair[3].point2.y;

	float a, b, c, d, e, f, g, h;

	a = -(u0*v0*v1*x2 - u0*v0*v2*x1 - u0*v0*v1*x3 + u0*v0*v3*x1 - u1*v0*v1*x2 + u1*v1*v2*x0 + u0*v0*v2*x3 - u0*v0*v3*x2 + u1*v0*v1*x3 - u1*v1*v3*x0 + u2*v0*v2*x1 - u2*v1*v2*x0
		- u1*v1*v2*x3 + u1*v1*v3*x2 - u2*v0*v2*x3 + u2*v2*v3*x0 - u3*v0*v3*x1 + u3*v1*v3*x0 + u2*v1*v2*x3 - u2*v2*v3*x1 + u3*v0*v3*x2 - u3*v2*v3*x0 - u3*v1*v3*x2 + u3*v2*v3*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	b = (u0*u1*v0*x2 - u0*u2*v0*x1 - u0*u1*v0*x3 - u0*u1*v1*x2 + u0*u3*v0*x1 + u1*u2*v1*x0 + u0*u1*v1*x3 + u0*u2*v0*x3 + u0*u2*v2*x1 - u0*u3*v0*x2 - u1*u2*v2*x0 - u1*u3*v1*x0
		- u0*u2*v2*x3 - u0*u3*v3*x1 - u1*u2*v1*x3 + u1*u3*v1*x2 + u1*u3*v3*x0 + u2*u3*v2*x0 + u0*u3*v3*x2 + u1*u2*v2*x3 - u2*u3*v2*x1 - u2*u3*v3*x0 - u1*u3*v3*x2 + u2*u3*v3*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	c = (u0*v1*x2 - u0*v2*x1 - u1*v0*x2 + u1*v2*x0 + u2*v0*x1 - u2*v1*x0 - u0*v1*x3 + u0*v3*x1 + u1*v0*x3 - u1*v3*x0 - u3*v0*x1 + u3*v1*x0
		+ u0*v2*x3 - u0*v3*x2 - u2*v0*x3 + u2*v3*x0 + u3*v0*x2 - u3*v2*x0 - u1*v2*x3 + u1*v3*x2 + u2*v1*x3 - u2*v3*x1 - u3*v1*x2 + u3*v2*x1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	d = (u0*u1*v0*v2*x3 - u0*u1*v0*v3*x2 - u0*u2*v0*v1*x3 + u0*u2*v0*v3*x1 + u0*u3*v0*v1*x2 - u0*u3*v0*v2*x1 - u0*u1*v1*v2*x3 + u0*u1*v1*v3*x2 + u1*u2*v0*v1*x3 - u1*u2*v1*v3*x0 - u1*u3*v0*v1*x2 + u1*u3*v1*v2*x0
		+ u0*u2*v1*v2*x3 - u0*u2*v2*v3*x1 - u1*u2*v0*v2*x3 + u1*u2*v2*v3*x0 + u2*u3*v0*v2*x1 - u2*u3*v1*v2*x0 - u0*u3*v1*v3*x2 + u0*u3*v2*v3*x1 + u1*u3*v0*v3*x2 - u1*u3*v2*v3*x0 - u2*u3*v0*v3*x1 + u2*u3*v1*v3*x0)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	e = -(u0*v0*v1*y2 - u0*v0*v2*y1 - u0*v0*v1*y3 + u0*v0*v3*y1 - u1*v0*v1*y2 + u1*v1*v2*y0 + u0*v0*v2*y3 - u0*v0*v3*y2 + u1*v0*v1*y3 - u1*v1*v3*y0 + u2*v0*v2*y1 - u2*v1*v2*y0
		- u1*v1*v2*y3 + u1*v1*v3*y2 - u2*v0*v2*y3 + u2*v2*v3*y0 - u3*v0*v3*y1 + u3*v1*v3*y0 + u2*v1*v2*y3 - u2*v2*v3*y1 + u3*v0*v3*y2 - u3*v2*v3*y0 - u3*v1*v3*y2 + u3*v2*v3*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	f = (u0*u1*v0*y2 - u0*u2*v0*y1 - u0*u1*v0*y3 - u0*u1*v1*y2 + u0*u3*v0*y1 + u1*u2*v1*y0 + u0*u1*v1*y3 + u0*u2*v0*y3 + u0*u2*v2*y1 - u0*u3*v0*y2 - u1*u2*v2*y0 - u1*u3*v1*y0
		- u0*u2*v2*y3 - u0*u3*v3*y1 - u1*u2*v1*y3 + u1*u3*v1*y2 + u1*u3*v3*y0 + u2*u3*v2*y0 + u0*u3*v3*y2 + u1*u2*v2*y3 - u2*u3*v2*y1 - u2*u3*v3*y0 - u1*u3*v3*y2 + u2*u3*v3*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	g = (u0*v1*y2 - u0*v2*y1 - u1*v0*y2 + u1*v2*y0 + u2*v0*y1 - u2*v1*y0 - u0*v1*y3 + u0*v3*y1 + u1*v0*y3 - u1*v3*y0 - u3*v0*y1 + u3*v1*y0
		+ u0*v2*y3 - u0*v3*y2 - u2*v0*y3 + u2*v3*y0 + u3*v0*y2 - u3*v2*y0 - u1*v2*y3 + u1*v3*y2 + u2*v1*y3 - u2*v3*y1 - u3*v1*y2 + u3*v2*y1)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	h = (u0*u1*v0*v2*y3 - u0*u1*v0*v3*y2 - u0*u2*v0*v1*y3 + u0*u2*v0*v3*y1 + u0*u3*v0*v1*y2 - u0*u3*v0*v2*y1 - u0*u1*v1*v2*y3 + u0*u1*v1*v3*y2 + u1*u2*v0*v1*y3 - u1*u2*v1*v3*y0 - u1*u3*v0*v1*y2 + u1*u3*v1*v2*y0
		+ u0*u2*v1*v2*y3 - u0*u2*v2*v3*y1 - u1*u2*v0*v2*y3 + u1*u2*v2*v3*y0 + u2*u3*v0*v2*y1 - u2*u3*v1*v2*y0 - u0*u3*v1*v3*y2 + u0*u3*v2*v3*y1 + u1*u3*v0*v3*y2 - u1*u3*v2*v3*y0 - u2*u3*v0*v3*y1 + u2*u3*v1*v3*y0)
		/ (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
			- u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);

	return HomographyMatrix(a, b, c, d, e, f, g, h);
}


CImg<unsigned char> Luminance(CImg<unsigned char> img) {
	int width = img._width;
	int height = img._height;
	CImg<unsigned char> gray(width, height, 1, 1);
	unsigned char gray_temp = 0;
	cimg_forXY(img, x, y) {
		gray_temp = 0.299 * img(x, y, 0) + 0.587 * img(x, y, 1) + 0.114 * img(x, y, 2);
		gray(x, y) = gray_temp;
	}
	return gray;
}

map<vector<float>, VlSiftKeypoint> get_sift_features(CImg<unsigned char> gray_img) {

	map<vector<float>, VlSiftKeypoint> features;

	vl_sift_pix *img_data = new vl_sift_pix[gray_img._width * gray_img._height];
	cimg_forXY(gray_img, x, y) {
		img_data[x + y * gray_img._width] = gray_img(x, y);
	}
	int noctaves = 4, nlevels = 5, o_min = 0;
	VlSiftFilt *SiftFilt = NULL;
	SiftFilt = vl_sift_new(gray_img._width, gray_img._height, noctaves, nlevels, o_min);
	if (vl_sift_process_first_octave(SiftFilt, img_data) != VL_ERR_EOF) {
		while (true) {
			//计算关键点
			vl_sift_detect(SiftFilt);
			//获取关键点
			VlSiftKeypoint const *keypoints = vl_sift_get_keypoints(SiftFilt);

			for (int i = 0; i < SiftFilt->nkeys; i++) {
				double angles[4];
				VlSiftKeypoint keypoint_pointer = *keypoints;

				//unsigned char red[] = { 255, 0, 0 };
				//img.draw_circle(keypoint_pointer.x, keypoint_pointer.y, keypoint_pointer.sigma, red, 1, 1);

				keypoints++;
				int angleCount = vl_sift_calc_keypoint_orientations(SiftFilt, angles, &keypoint_pointer); //计算方向
				for (int j = 0; j < angleCount; j++) {
					vl_sift_pix descr[128];
					vl_sift_calc_keypoint_descriptor(SiftFilt, descr, &keypoint_pointer, angles[j]);
					vector<float> descriptor;
					for (int k = 0; k < 128; k++) {
						descriptor.push_back(descr[k]);
					}

					features.insert(pair<vector<float>, VlSiftKeypoint>(descriptor, keypoint_pointer));
				}
			}
			if (vl_sift_process_next_octave(SiftFilt) == VL_ERR_EOF) break;
		}
	}
	vl_sift_delete(SiftFilt);
	delete img_data;
	img_data = NULL;
	return features;
}

vector<keypointpair> find_2_nearest_neighbours(map<vector<float>, VlSiftKeypoint> fea1, map<vector<float>, VlSiftKeypoint> fea2) {
	//存储配对好的keypoint
	vector<keypointpair> result;

	int len = fea1.size();
	int dim = 128;
	int num_of_trees = 1;

	VlKDForest *kdforest = vl_kdforest_new(VL_TYPE_FLOAT, dim, num_of_trees, VlDistanceL1);

	//取出全部数据
	float *data = new float[dim * len];
	map<vector<float>, VlSiftKeypoint>::iterator iter;
	int i = 0;
	for (i = 0, iter = fea1.begin(); iter != fea1.end(); iter ++, i ++) {
		vector<float> des = iter->first;
		for (int j = 0; j < dim; j++) {
			data[j + dim * i] = des[j];
		}
	}

	vl_kdforest_build(kdforest, len, data);

	VlKDForestSearcher* searcher = vl_kdforest_new_searcher(kdforest);

	VlKDForestNeighbor neighbors[2];
	for (iter = fea2.begin(); iter != fea2.end(); iter++) {
		float search_point[128];
		for (int i = 0; i < dim; i++) {
			search_point[i] = (iter->first)[i];
		}
		vl_kdforestsearcher_query(searcher, neighbors, 2, search_point);

		if (neighbors[0].distance / neighbors[1].distance < 0.5) {
			vector<float> temp_data;
			for (int i = 0; i < dim; i++) {
				temp_data.push_back(data[i + neighbors[0].index * dim]);
			}
			VlSiftKeypoint p1 = fea1.find(temp_data)->second;
			VlSiftKeypoint p2 = iter->second;
			result.push_back(keypointpair(p1, p2));
		}
	}
	vl_kdforestsearcher_delete(searcher);
	vl_kdforest_delete(kdforest);
	delete[]data;
	data = NULL;
	return result;
}

void mixImageAndDrawPairLines(CImg<unsigned char> img1, CImg<unsigned char> img2, vector<keypointpair> pairs) {
	CImg<unsigned char> img = CImg<unsigned char>(img1._width + img2._width, Max(img1._height, img2._height), 1, 3, 0);

	cimg_forXY(img, x, y) {
		if (x < img1._width) {
			if (y < img1._height) {
				img(x, y, 0) = img1(x, y, 0);
				img(x, y, 1) = img1(x, y, 1);
				img(x, y, 2) = img1(x, y, 2);
			}
		}
		else {
			if (y < img2._height) {
				img(x, y, 0) = img2(x, y, 0);
				img(x, y, 1) = img2(x, y, 1);
				img(x, y, 2) = img2(x, y, 2);
			}
		}
	}

	const double blue[] = { 0, 255, 255 };
	for (int i = 0; i < pairs.size(); i++) {
		int x1 = pairs[i].point1.x;
		int y1 = pairs[i].point1.y;

		int x2 = pairs[i].point2.x + img1._width;
		int y2 = pairs[i].point2.y;

		img.draw_line(x1, y1, x2, y2, blue);
	}
	img.display();
}

void mixImageAndDrawRealPairLines(CImg<unsigned char> img1, CImg<unsigned char> img2, vector<keypointpair> pairs, vector<int> real_pairs) {
	CImg<unsigned char> img = CImg<unsigned char>(img1._width + img2._width, Max(img1._height, img2._height), 1, 3, 0);

	cimg_forXY(img, x, y) {
		if (x < img1._width) {
			if (y < img1._height) {
				img(x, y, 0) = img1(x, y, 0);
				img(x, y, 1) = img1(x, y, 1);
				img(x, y, 2) = img1(x, y, 2);
			}
		}
		else {
			if (y < img2._height) {
				img(x, y, 0) = img2(x, y, 0);
				img(x, y, 1) = img2(x, y, 1);
				img(x, y, 2) = img2(x, y, 2);
			}
		}
	}

	const double blue[] = { 0, 255, 255 };
	for (int i = 0; i < real_pairs.size(); i++) {
		int index = real_pairs[i];
		int x1 = pairs[index].point1.x;
		int y1 = pairs[index].point1.y;

		int x2 = pairs[index].point2.x + img1._width;
		int y2 = pairs[index].point2.y;

		img.draw_line(x1, y1, x2, y2, blue);
	}
	img.display();
}

HomographyMatrix RANSAC(vector<keypointpair>& pairs) {
	int s = 4;  //计算点的数目
	float p = 0.99;
	float e = 0.5;

	//long N = ceil(log(1 - p) / log(1 - pow(1 - e, s)));
	int N = 10000;
	vector<int> index_of_inliers, max_inliers;
	float mindis = 1000000;
	while (N--) {
		srand(time(0));  //设置随机种子

		//随机选取4对点
		vector<keypointpair> random_four_pairs;
		set<int> check_duplicate;
		while (random_four_pairs.size() < 4) {
			int index = rand() % pairs.size();
			if (check_duplicate.find(index) == check_duplicate.end()) {
				check_duplicate.insert(index);
				random_four_pairs.push_back(pairs[index]);
			}
		}

		//计算变换矩阵
		/*MatrixXf UV(8, 1);
		UV << random_four_pairs[0].point2.x, random_four_pairs[1].point2.x,
			random_four_pairs[2].point2.x, random_four_pairs[3].point2.x,
			random_four_pairs[0].point2.y, random_four_pairs[1].point2.y,
			random_four_pairs[2].point2.y, random_four_pairs[3].point2.y;

		MatrixXf A(8, 8);
		A << random_four_pairs[0].point1.x, random_four_pairs[0].point1.y, 1, 0, 0, 0, -(random_four_pairs[0].point1.x * random_four_pairs[0].point2.x), -random_four_pairs[0].point1.y * random_four_pairs[0].point2.x,
			random_four_pairs[1].point1.x, random_four_pairs[1].point1.y, 1, 0, 0, 0, -(random_four_pairs[1].point1.x * random_four_pairs[1].point2.x), -random_four_pairs[1].point1.y * random_four_pairs[1].point2.x,
			random_four_pairs[2].point1.x, random_four_pairs[2].point1.y, 1, 0, 0, 0, -(random_four_pairs[2].point1.x * random_four_pairs[2].point2.x), -random_four_pairs[2].point1.y * random_four_pairs[2].point2.x,
			random_four_pairs[3].point1.x, random_four_pairs[3].point1.y, 1, 0, 0, 0, -(random_four_pairs[3].point1.x * random_four_pairs[3].point2.x), -random_four_pairs[3].point1.y * random_four_pairs[3].point2.x,
			0, 0, 0, random_four_pairs[0].point1.x, random_four_pairs[0].point1.y, 1, -(random_four_pairs[0].point1.x * random_four_pairs[0].point2.y), -random_four_pairs[0].point1.y * random_four_pairs[0].point2.y,
			0, 0, 0, random_four_pairs[1].point1.x, random_four_pairs[1].point1.y, 1, -(random_four_pairs[1].point1.x * random_four_pairs[1].point2.y), -random_four_pairs[1].point1.y * random_four_pairs[1].point2.y,
			0, 0, 0, random_four_pairs[2].point1.x, random_four_pairs[2].point1.y, 1, -(random_four_pairs[2].point1.x * random_four_pairs[2].point2.y), -random_four_pairs[2].point1.y * random_four_pairs[2].point2.y,
			0, 0, 0, random_four_pairs[3].point1.x, random_four_pairs[3].point1.y, 1, -(random_four_pairs[3].point1.x * random_four_pairs[3].point2.y), -random_four_pairs[3].point1.y * random_four_pairs[3].point2.y;

		MatrixXf H(8, 1);
		H = A.inverse() * UV;*/

		HomographyMatrix H = get_homography_matrix(random_four_pairs);

		index_of_inliers.clear(); //清空inliers
		for (int i = 0; i < pairs.size(); i++) {
			if (check_duplicate.find(i) != check_duplicate.end())
				continue;

			float x1 = pairs[i].point1.x;
			float y1 = pairs[i].point1.y;
			float x2 = pairs[i].point2.x;
			float y2 = pairs[i].point2.y;

			float warpx = H.a * x1 + H.b * y1 + H.c * x1 * y1 + H.d;
			float warpy = H.e * x1 + H.f * y1 + H.g * x1 * y1 + H.h;

			//float wrapx = ((x2 * H(0) + y2 * H(1) + H(3))/(H(6) * x2 + H(7) * y2 + 1));
			//float wrapy = ((x2 * H(3) + y2 * H(4) + H(5))/(H(6) * x2 + H(7) * y2 + 1));

			//float wrapx = x1 * H(0) + y1 * H(1) + x1 * y1 * H(2) + H(3);
			//float wrapy = x1 * H(4) + y1 * H(5) + x1 * y1 * H(6) + H(7);
			

			
			float dist = sqrt((warpx - x2) * (warpx - x2) + (warpy - y2) * (warpy - y2));
			if (dist < mindis) mindis = dist;
			if (dist < 1)
				index_of_inliers.push_back(i);
		}
		if (max_inliers.size() < index_of_inliers.size()) {
			max_inliers = index_of_inliers;
		}
	}
	cout << max_inliers.size() << endl;
	//cout << "min:" << mindis << endl;
	//return max_inliers;

	int inliers_num = max_inliers.size();
	assert(inliers_num > 0);
	CImg<double> A(4, inliers_num, 1, 1, 0);
	CImg<double> bx(1, inliers_num, 1, 1, 0);
	CImg<double> by(1, inliers_num, 1, 1, 0);
	for (int i = 0; i < inliers_num; ++i) {
		int idx = max_inliers[i];
		A(0, i) = pairs[idx].point1.x;
		A(1, i) = pairs[idx].point1.y;
		A(2, i) = pairs[idx].point1.x * pairs[idx].point1.y;
		A(3, i) = 1;

		bx(0, i) = pairs[idx].point2.x;
		by(0, i) = pairs[idx].point2.y;
	}
	CImg<double> x1 = bx.get_solve(A); // solve Ax=B
	CImg<double> x2 = by.get_solve(A);

	return HomographyMatrix(x1(0, 0), x1(0, 1), x1(0, 2),
		x1(0, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3));
}

float get_warped_x(float x, float y, HomographyMatrix H) {
	return H.a * x + H.b * y + H.c * x * y + H.d;
}

float get_warped_y(float x, float y, HomographyMatrix H) {
	return H.e * x + H.f * y + H.g * x * y + H.h;
}

float get_min_warped_x(const CImg<unsigned char> &input_img, HomographyMatrix H) {
	int w = input_img.width();
	int h = input_img.height();
	float min_x = cimg::min(get_warped_x(0, 0, H), get_warped_x(w - 1, 0, H),
		get_warped_x(0, h - 1, H), get_warped_x(w - 1, h - 1, H));
	return min_x < 0 ? min_x : 0;
}

float get_min_warped_y(const CImg<unsigned char> &input_img, HomographyMatrix H) {
	int w = input_img.width();
	int h = input_img.height();
	float min_y = cimg::min(get_warped_y(0, 0, H), get_warped_y(w - 1, 0, H),
		get_warped_y(0, h - 1, H), get_warped_y(w - 1, h - 1, H));
	return min_y < 0 ? min_y : 0;
}

float get_max_warped_x(const CImg<unsigned char> &input_img, HomographyMatrix H,
	const CImg<unsigned char> &stitched_img) {
	int w = input_img.width();
	int h = input_img.height();
	float max_x = cimg::max(get_warped_x(0, 0, H), get_warped_x(w - 1, 0, H),
		get_warped_x(0, h - 1, H), get_warped_x(w - 1, h - 1, H));
	return max_x >= stitched_img.width() ? max_x : stitched_img.width();
}

float get_max_warped_y(const CImg<unsigned char> &input_img, HomographyMatrix H,
	const CImg<unsigned char> &stitched_img) {
	int w = input_img.width();
	int h = input_img.height();
	float max_y = cimg::max(get_warped_y(0, 0, H), get_warped_y(w - 1, 0, H),
		get_warped_y(0, h - 1, H), get_warped_y(w - 1, h - 1, H));
	return max_y >= stitched_img.height() ? max_y : stitched_img.height();
}

void img_homography_warping(const CImg<unsigned char> &src,
	CImg<unsigned char> &dst, HomographyMatrix H,
	float offset_x, float offset_y) {
	// inverse warping
	cimg_forXY(dst, x, y) {
		float warped_x = get_warped_x(x + offset_x, y + offset_y, H);
		float warped_y = get_warped_y(x + offset_x, y + offset_y, H);
		if (warped_x >= 0 && warped_x < src.width() &&
			warped_y >= 0 && warped_y < src.height()) {
			cimg_forC(dst, c) {
				dst(x, y, c) = src(floor(warped_x), floor(warped_y), c);
			}
		}
	}
}

void img_shift(const CImg<unsigned char> &src,
	CImg<unsigned char> &dst, float offset_x, float offset_y) {
	cimg_forXY(dst, x, y) {
		int x0 = x + offset_x;
		int y0 = y + offset_y;
		if (x0 >= 0 && x0 < src.width() &&
			y0 >= 0 && y0 < src.height()) {
			cimg_forC(dst, c) {
				dst(x, y, c) = src(x0, y0, c);
			}
		}
	}
}

void feature_homography_warping(map<vector<float>, VlSiftKeypoint> &feature,
	HomographyMatrix H, float offset_x, float offset_y) {
	map<vector<float>, VlSiftKeypoint>::iterator it = feature.begin();
	for (it; it != feature.end(); ++it) {
		float px = it->second.x; // coordinate
		float py = it->second.y;
		it->second.x = get_warped_x(px, py, H) - offset_x;
		it->second.y = get_warped_y(px, py, H) - offset_y;
		it->second.ix = int(it->second.x); // unormolized coordinate
		it->second.iy = int(it->second.y);
	}
}

void feature_shift(map<vector<float>, VlSiftKeypoint> &feature,
	float offset_x, float offset_y) {
	map<vector<float>, VlSiftKeypoint>::iterator it = feature.begin();
	for (it; it != feature.end(); ++it) {
		it->second.x -= offset_x; // coordinate
		it->second.y -= offset_y;
		it->second.ix = int(it->second.x); // unormolized coordinate
		it->second.iy = int(it->second.y);
	}
}

CImg<unsigned char> multiband_blending(const CImg<unsigned char> &a, const CImg<unsigned char> &b) {
	int w = a.width(), h = a.height();

	int sum_a_x = 0, sum_a_y = 0;
	int width_mid_a = 0;

	int sum_overlap_x = 0, sum_overlap_y = 0;
	int width_mid_overlap = 0;

	int mid_y = h / 2;
	int x = 0;
	while (a(x, mid_y) == 0) ++x; 
	for (x; x < w; ++x) {
		if (a(x, mid_y) != 0) { 
			sum_a_x += x;
			++width_mid_a;
			if (b(x, mid_y) != 0) {
				sum_overlap_x += x;
				++width_mid_overlap;
			}
		}
	}

	int max_len = w >= h ? w : h;
	int level_num = floor(log2(max_len));

	vector<CImg<float>> a_pyramid(level_num);
	vector<CImg<float>> b_pyramid(level_num);
	vector<CImg<float>> mask(level_num);

	a_pyramid[0] = a;
	b_pyramid[0] = b;
	mask[0] = CImg<float>(w, h, 1, 1, 0);
	assert(width_mid_a > 0);
	assert(width_mid_overlap > 0);
	float ratio = 1.0 * sum_a_x / width_mid_a;
	float overlap_ratio = 1.0 * sum_overlap_x / width_mid_overlap;
	
	if (ratio < overlap_ratio) {
		for (int x = 0; x < overlap_ratio; ++x)
			for (int y = 0; y < h; ++y)
				mask[0](x, y) = 1;
	}
	else {
		for (int x = overlap_ratio + 1; x < w; ++x)
			for (int y = 0; y < h; ++y)
				mask[0](x, y) = 1;
	}

	for (int i = 1; i < level_num; ++i) {
		int wp = a_pyramid[i - 1].width() / 2;
		int hp = a_pyramid[i - 1].height() / 2;
		int sp = a_pyramid[i - 1].spectrum();
		a_pyramid[i] = a_pyramid[i - 1].get_blur(2, true, true).
			get_resize(wp, hp, 1, sp, 3);
		b_pyramid[i] = b_pyramid[i - 1].get_blur(2, true, true).
			get_resize(wp, hp, 1, sp, 3);
		mask[i] = mask[i - 1].get_blur(2, true, true).
			get_resize(wp, hp, 1, sp, 3);
	}
 
	for (int i = 0; i < level_num - 1; ++i) {
		int wp = a_pyramid[i].width();
		int hp = a_pyramid[i].height();
		int sp = a_pyramid[i].spectrum();
		a_pyramid[i] -= a_pyramid[i + 1].get_resize(wp, hp, 1, sp, 3);
		b_pyramid[i] -= b_pyramid[i + 1].get_resize(wp, hp, 1, sp, 3);
	}

	vector<CImg<float>> blend_pyramid(level_num);
	for (int i = 0; i < level_num; ++i) {
		blend_pyramid[i] = CImg<float>(a_pyramid[i].width(),
			a_pyramid[i].height(), 1, a_pyramid[i].spectrum(), 0);
		cimg_forXYC(blend_pyramid[i], x, y, c) {
			blend_pyramid[i](x, y, c) =
				a_pyramid[i](x, y, c) * mask[i](x, y) +
				b_pyramid[i](x, y, c) * (1.0 - mask[i](x, y));
		}
	}

	CImg<float> expand = blend_pyramid[level_num - 1];
	for (int i = level_num - 2; i >= 0; --i) {
		expand.resize(blend_pyramid[i].width(),
			blend_pyramid[i].height(), 1, blend_pyramid[i].spectrum(), 3);
		cimg_forXYC(blend_pyramid[i], x, y, c) {
			expand(x, y, c) = blend_pyramid[i](x, y, c) + expand(x, y, c);
			if (expand(x, y, c) > 255) expand(x, y, c) = 255;
			else if (expand(x, y, c) < 0) expand(x, y, c) = 0;
		}
	}
	return expand;
}


template <class T>
T bilinear_interpolation(const CImg<T>& image, float x, float y, int channel) {
	/* This function comes from https://github.com/AmazingZhen/ImageStitching */
	assert(x >= 0 && x < image.width());
	assert(y >= 0 && y < image.height());
	assert(channel <= image.spectrum());

	int x_pos = floor(x);
	float x_u = x - x_pos;
	int xb = (x_pos < image.width() - 1) ? x_pos + 1 : x_pos;

	int y_pos = floor(y);
	float y_v = y - y_pos;
	int yb = (y_pos < image.height() - 1) ? y_pos + 1 : y_pos;

	float P1 = image(x_pos, y_pos, channel) * (1 - x_u) + image(xb, y_pos, channel) * x_u;
	float P2 = image(x_pos, yb, channel) * (1 - x_u) + image(xb, yb, channel) * x_u;

	return P1 * (1 - y_v) + P2 * y_v;
}

CImg<unsigned char> cylinderProjection(const CImg<unsigned char> &src) {
	/* This function comes from https://github.com/AmazingZhen/ImageStitching */
	int projection_width, projection_height;
	CImg<unsigned char> res(src.width(), src.height(), 1, src.spectrum(), 0);
	float r;
	float angle = 15.0;
	if (src.width() > src.height()) {
		projection_width = src.height();
		projection_height = src.width();

		r = (projection_width / 2.0) / tan(angle * cimg::PI / 180.0);

		for (int i = 0; i < res.width(); i++) {
			for (int j = 0; j < res.height(); j++) {
				float dst_x = j - projection_width / 2;
				float dst_y = i - projection_height / 2;

				float k = r / sqrt(r * r + dst_x * dst_x);
				float src_x = dst_x / k;
				float src_y = dst_y / k;

				if (src_x + projection_width / 2 >= 0 && src_x + projection_width / 2 < src.height()
					&& src_y + projection_height / 2 >= 0 && src_y + projection_height / 2 < src.width()) {
					for (int k = 0; k < res.spectrum(); k++) {
						res(i, j, k) = bilinear_interpolation(src, src_y + projection_height / 2, src_x + projection_width / 2, k);
					}
				}
			}
		}

	}
	else {
		projection_width = src.width();
		projection_height = src.height();

		r = (projection_width / 2.0) / tan(angle * cimg::PI / 180.0);

		for (int i = 0; i < res.width(); i++) {
			for (int j = 0; j < res.height(); j++) {
				float dst_x = i - projection_width / 2;
				float dst_y = j - projection_height / 2;

				float k = r / sqrt(r * r + dst_x * dst_x);
				float src_x = dst_x / k;
				float src_y = dst_y / k;

				if (src_x + projection_width / 2 >= 0 && src_x + projection_width / 2 < src.width()
					&& src_y + projection_height / 2 >= 0 && src_y + projection_height / 2 < src.height()) {
					for (int k = 0; k < res.spectrum(); k++) {
						res(i, j, k) = bilinear_interpolation(src, src_x + projection_width / 2, src_y + projection_height / 2, k);
					}
				}
			}
		}

	}

	return res;
}