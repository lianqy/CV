#include "stdafx.h"
#include <iostream>
#include "CImg.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "utils.h"
#include <map>

using namespace std;
using namespace cimg_library;

int main() {
	CImg<unsigned char> img[4];
	img[0].load_bmp("./TEST-ImageData/pano1_0008.bmp");
	img[1].load_bmp("./TEST-ImageData/pano1_0009.bmp");
	img[2].load_bmp("./TEST-ImageData/pano1_0010.bmp");
	img[3].load_bmp("./TEST-ImageData/pano1_0011.bmp");

	for (int i = 0; i < 4; i++) {
		img[i] = cylinderProjection(img[i]);
	}
	
	CImg<unsigned char> grayimg[4];
	for (int i = 0; i < 4; i++) {
		grayimg[i] = Luminance(img[i]);
	}
	
	//¼ì²âSIFTÌØÕ÷
	unsigned char red[] = { 255, 0, 0 };
	map<vector<float>, VlSiftKeypoint> features[4];

	for (int i = 0; i < 4; i++) {
		features[i] = get_sift_features(grayimg[i]);
	}

	//features[0] = get_sift_features(grayimg[0]);
	//features[1] = get_sift_features(grayimg[1]);
	//cout << features[0].size() << endl;
	//cout << features[1].size() << endl;
	//map<vector<float>, VlSiftKeypoint>::iterator iter;
	/*for (int i = 0; i < 2; i++) {
		for (iter = features[i].begin(); iter != features[i].end(); iter++) {
			int x = iter->second.x;
			int y = iter->second.y;
			float r = (iter->second.sigma) / 2;
			img[i].draw_circle(x, y, (int)r, red, 1, 1);
		}
		//img[i].display();
	}*/

	CImg<unsigned char> stitched_img = img[0];

	for (int i = 0; i < 3; i++) {
		vector<keypointpair> fea_pairs = find_2_nearest_neighbours(features[i], features[i + 1]);
		//cout << fea_pairs.size() << endl;
		//mixImageAndDrawPairLines(img[0], img[1], fea_pairs);

		HomographyMatrix H = RANSAC(fea_pairs);

		vector<keypointpair> feature_pairs_inv;
		for (int j = 0; j < fea_pairs.size(); ++j) {
			feature_pairs_inv.push_back(
				keypointpair(fea_pairs[j].point2, fea_pairs[j].point1));
		}

		HomographyMatrix H_inv = RANSAC(feature_pairs_inv);
		//mixImageAndDrawRealPairLines(img[0], img[1], fea_pairs, real_pairs);
		//cout << real_pairs.size() << endl;

		CImg<unsigned char> adjacent_img(img[i + 1]);

		float min_x = get_min_warped_x(adjacent_img, H_inv); // min <= 0
		float min_y = get_min_warped_y(adjacent_img, H_inv);
		float max_x = get_max_warped_x(adjacent_img, H_inv, stitched_img);
		float max_y = get_max_warped_y(adjacent_img, H_inv, stitched_img);

		int out_w = ceil(max_x - min_x);
		int out_h = ceil(max_y - min_y);

		CImg<unsigned char> last_stitch(out_w, out_h, 1, adjacent_img.spectrum(), 0);
		CImg<unsigned char> next_stitch(out_w, out_h, 1, adjacent_img.spectrum(), 0);

		img_shift(stitched_img, last_stitch, min_x, min_y);
		img_homography_warping(adjacent_img, next_stitch, H, min_x, min_y);
		feature_homography_warping(features[i + 1], H_inv, min_x, min_y);
		feature_shift(features[i], min_x, min_y);

		last_stitch.display();
		next_stitch.display();

		stitched_img = multiband_blending(last_stitch, next_stitch);
		//stitched_img = cylinderProjection(stitched_img);
		stitched_img.display();
	}
	stitched_img.get_crop(100, 100, stitched_img._width - 80, stitched_img._height - 80);
	stitched_img.display();

	system("pause");
	return 0;
}