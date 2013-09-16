/*
 * utils.hpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;
using namespace cv;
using namespace boost::filesystem;

void loadImages(string, vector<Mat>&, float);
void createFolds(vector<Mat>&, vector<vector<Mat> >&, int);
float euclideanDistance(Mat, Mat);
void patcher(Mat, int, int, vector<vector<Mat> >&);
void myPCA(vector<Mat> &trainingSet, Mat &vecs, Mat &valsDiag);

#endif /* UTILS_HPP_ */
