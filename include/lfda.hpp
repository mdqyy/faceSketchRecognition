#ifndef __LFDA_HPP__
#define __LFDA_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include "descriptors.hpp"

using namespace std;
using namespace cv;

class LFDA
{
private:
  vector<Mat> trainingPhotos,	trainingSketches, Xsk, Xpk, Xk, omegaK;
  int size, overlap;
  
public:
  LFDA(vector<Mat>& trainingPhotos,vector<Mat>& trainingSketches, int size, int overlap);
  virtual ~LFDA();
  void compute();
  Mat project(Mat image);
  vector<Mat> extractDescriptors(Mat img, int size, int delta);
};

#endif
