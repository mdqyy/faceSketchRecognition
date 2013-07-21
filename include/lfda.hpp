/*
 * lfda.hpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#ifndef LFDA_HPP_
#define LFDA_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/features2d.hpp> //SURF/SIFT
//#include <subspace.hpp>
//#include <facerec.hpp>
//#include <decomposition.hpp>
//#include <helper.hpp>
//#include <lbp.hpp>

using namespace cv;
using namespace std;

void patcher(Mat, int, int, vector<vector<Mat> >&);
void calcLBPHistogram(Mat, Mat&);
void calcSIFTDescriptors(Mat, Mat&);

#endif /* LFDA_HPP_ */
