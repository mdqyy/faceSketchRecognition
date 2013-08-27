/*
 * eigenTransformation.hpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#ifndef EIGENTRANSFORMATION_HPP_
#define EIGENTRANSFORMATION_HPP_

#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include <vector>

using namespace cv;
using namespace Eigen;

void createEigenSpace(vector<Mat>&, Mat&, Mat&, Mat&, Mat&, Mat&);

#endif /* EIGENTRANSFORMATION_HPP_ */
