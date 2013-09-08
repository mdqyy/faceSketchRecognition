#ifndef __DESCRIPTORS_HPP__
#define __DESCRIPTORS_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/features2d.hpp> //SURF/SIFT
//#include "helper.hpp"

using namespace cv;
using namespace std;


void elbp(InputArray src, OutputArray dst, int radius=1, int neighbors=8);

Mat elbp(InputArray src, int radius=1, int neighbors=8);

void calcLBPHistogram(Mat, Mat&);
void calcSIFTDescriptors(Mat, Mat&);



#endif
