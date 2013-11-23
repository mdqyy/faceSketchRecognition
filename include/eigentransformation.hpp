#ifndef __EIGENTRANSFORMATION_HPP__
#define __EIGENTRANSFORMATION_HPP__

#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace Eigen;

class Eigentransformation
{
  
private:
  Mat avgPhoto, avgSketch, eigenPhotos, eigenSketches,
  pPhotos, pSketches, vecsPhotos, vecsSketches, valsDiagPhotos, valsDiagSketches;
  vector<Mat> trainingPhotos,	trainingSketches;
  int nTraining;
  
  void createEigenSpace(vector<Mat>&, Mat&, Mat&, Mat&, Mat&, Mat&);
  
public:
  Eigentransformation(vector<Mat>& trainingPhotos,vector<Mat>& trainingSketches);
  virtual ~Eigentransformation();
  void compute();
  void projectPhoto(Mat& photo, Mat& photoB, Mat& photoContr, Mat& recSketchB);
  void projectSketch(Mat& sketch, Mat& sketchB, Mat& sketchContr, Mat& recPhotoB);
};

#endif
