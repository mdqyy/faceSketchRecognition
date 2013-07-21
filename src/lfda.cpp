/*
 * lfda.cpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#include "lfda.hpp"
#include "lbp.hpp"

int  UniformPattern59[256] = {
		1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
		12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
		17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
		23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
		30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
		37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
		43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
		48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58
};

void patcher(Mat img, int size, int delta, vector<vector<Mat> > &result){
	int w = img.cols, h=img.rows;

	for(int i=0;i<=h-size;i+=(size-delta)){
		vector<Mat> col;
		for(int j=0;j<=w-size;j+=(size-delta)){
			col.push_back(img(Rect(j,i,size,size)));
		}
		result.push_back(col);
	}

}

void calcLBPHistogram(Mat src, Mat &hist){
	hist = Mat::zeros(1, 59*4, CV_32F);
	Mat temp;
	for(int k=1; k<8; k+=2){
		temp = elbp(src,k,8);
		for(int i = 0; i < temp.rows; i++) {
			for(int j = 0; j < temp.cols; j++){
				int bin = UniformPattern59[temp.at<int>(i,j)];
				hist.at<float>(bin+((k-1)/2)*59) += 1;
			}
		}
	}
}

void calcSIFTDescriptors(Mat src, Mat &descriptors){
	//FeatureDetector* featureDetector = new SiftFeatureDetector();
	DescriptorExtractor* descriptorExtractor = new SiftDescriptorExtractor();
	//#if CV_MAJOR_VERSION*100+CV_MINOR_VERSION*10+CV_SUBMINOR_VERSION > 233
	//In OpenCV 2.4, default for SURF is extended version (of size 128)
	//((SurfFeatureDetector *)featureDetector)->extended = false;
	//((SurfDescriptorExtractor *)descriptorExtractor)->extended = false;
	//#endif

	//Mat mask(0, 0, CV_8UC1);
	//Mat descriptors;
	vector<KeyPoint> keypoints;
	//featureDetector->detect(im, keypoints, mask);

	keypoints.push_back(KeyPoint(src.rows/2,src.cols/2,8));
	descriptorExtractor->compute(src, keypoints, descriptors);

	//cerr<<" descriptors.rows="<< descriptors.rows <<" keypoints.size()="
	//		<<keypoints.size()<<" descriptors.cols=" << descriptors.cols << endl;
}

