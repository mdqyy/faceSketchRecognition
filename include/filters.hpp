#ifndef FILTERS_HPP_
#define FILTERS_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;

Mat DoGFilter(Mat);
Mat GaussianFilter(Mat);
Mat CSDNFilter(Mat);

#endif