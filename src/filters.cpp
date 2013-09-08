#include "filters.hpp"

Mat DoGFilter(Mat src){
  
  Mat g1, g2, dst;
  GaussianBlur(src, g1, Size(5,5), 0);
  GaussianBlur(src, g2, Size(9,9), 0);
  
  dst = g2 - g1;
  normalize(dst,dst,0, 255, NORM_MINMAX, CV_8U);
  
  return dst;
}

Mat GaussianFilter(Mat src){
  
  Mat dst;
  GaussianBlur(src, dst, Size(5,5), 0);
  
  return dst;
}

Mat CSDNFilter(Mat src){
  
  Mat dst;
  src.convertTo(src, CV_32F);
  
  blur(src, dst, Size(16,16));
  divide(src, dst, dst);
  normalize(dst,dst,0, 255, NORM_MINMAX, CV_8U);
  
  return dst;
  
}