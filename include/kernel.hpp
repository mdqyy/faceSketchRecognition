#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
#include "descriptors.hpp"
#include "filters.hpp"

using namespace std;
using namespace cv;

class Kernel
{
private:
  vector<Mat> trainingPhotos,	trainingSketches, trainingPhotosDescriptors, trainingSketchesDescriptors;
  int patches;
  string filter, descriptor;
  Mat Kp, Kg, R, T2, mean;
  Mat projectGalleryIntern(Mat image);
  Mat projectProbeIntern(Mat image);
public:
  PCA pca;
  LDA lda;
  Kernel(vector<Mat>& trainingPhotos,vector<Mat>& trainingSketches, int patches, string filter, string descriptor);
  virtual ~Kernel();
  void compute();
  Mat projectGallery(Mat image);
  Mat projectProbe(Mat image);
  Mat extractDescriptors(Mat img);
  float calcKernel(Mat x, Mat y);
};

#endif
