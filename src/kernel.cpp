/*
 *    <one line to give the program's name and a brief idea of what it does.>
 *    Copyright (C) 2013  <copyright holder> <email>
 * 
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 * 
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 * 
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "kernel.hpp"

Kernel::Kernel(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches, int patches, string filter, string descriptor)
{
  this->trainingPhotos = trainingPhotos;
  this->trainingSketches = trainingSketches;
  this->patches = patches;
  this->filter = filter;
  this->descriptor = descriptor;
}

Kernel::~Kernel()
{
  
}

void Kernel::compute()
{
  
  int n = trainingPhotos.size();
  
  for(int i=0; i<n; i++){
    trainingPhotosDescriptors.push_back(extractDescriptors(trainingPhotos[i]));
    trainingSketchesDescriptors.push_back(extractDescriptors(trainingSketches[i]));
  }
  
  Kp = Mat::zeros(n/2,n/2,CV_32F);
  Kg = Mat::zeros(n/2,n/2,CV_32F);
  
  for(int i=0; i<n/2; i++)
    for(int j=0; j<n/2; j++){
      Kg.at<float>(i,j) = this->calcKernel(trainingPhotosDescriptors[i], trainingPhotosDescriptors[j]);
      Kp.at<float>(i,j) = this->calcKernel(trainingSketchesDescriptors[i],trainingSketchesDescriptors[j]);
    }
    
    R = Kg*((Kp).t()*Kp).inv()*(Kp).t();
  
  vector<int> _classes;
  
  for(int i=n/2; i<n; i++){
    if(T2.empty()){
      T2.push_back(this->projectProbeIntern(trainingSketchesDescriptors[i]));
      hconcat(T2,projectGalleryIntern(trainingPhotosDescriptors[i]),T2);
    }
    else{
      hconcat(T2,projectProbeIntern(trainingSketchesDescriptors[i]),T2);
      hconcat(T2,projectGalleryIntern(trainingPhotosDescriptors[i]),T2);
    }
    _classes.push_back(i);
    _classes.push_back(i);
  }
  
  this->pca.computeVar(T2, Mat(), CV_PCA_DATA_AS_COL, 0.99);
  this->mean = pca.mean.clone();
  //cout << mean.size() << endl;
  //cout << pca.eigenvectors.size() << endl;
  //cout << T2.size() << endl;
  Mat T2_pca = pca.eigenvectors*T2;
  T2_pca = T2_pca.t();
  //cout << T2_pca.size() << endl; 
  lda.compute(T2_pca, _classes);
  //cout << lda.eigenvectors().size() << endl;
}

Mat Kernel::projectGalleryIntern(Mat desc)
{
  int n = trainingPhotosDescriptors.size();
  Mat result = Mat::zeros(1,n/2,CV_32F);
  for(int i=0; i<n/2; i++)
    result.at<float>(i) = this->calcKernel(desc,trainingPhotosDescriptors[i]);
  
  //normalize(result,result,1,0,NORM_MINMAX, CV_32F);
    
  return result.t();
}

Mat Kernel::projectProbeIntern(Mat desc)
{
  int n = trainingSketchesDescriptors.size();
  Mat result = Mat::zeros(1,n/2,CV_32F);
  for(int i=0; i<n/2; i++)
    result.at<float>(i) = this->calcKernel(desc,trainingSketchesDescriptors[i]);
  
  result =  R*result.t();
  //normalize(result,result,1,0,NORM_MINMAX, CV_32F);
  
  return result;
}

Mat Kernel::projectGallery(Mat image){
  Mat desc = extractDescriptors(image);
  Mat temp = lda.eigenvectors().t();
  temp.convertTo(temp, CV_32F);
  Mat result = (temp*pca.eigenvectors)*(projectGalleryIntern(desc)-this->mean);
  return result;
}

Mat Kernel::projectProbe(Mat image){
  Mat desc = extractDescriptors(image);
  Mat temp = lda.eigenvectors().t();
  temp.convertTo(temp, CV_32F);
  Mat result = (temp*pca.eigenvectors)*(projectProbeIntern(desc)-this->mean);
  return result;
}

Mat Kernel::extractDescriptors(Mat image){
  int w = image.cols, h=image.rows, size=32, delta=16;
  Mat result, temp, img;
  int i = (patches/14)*delta, j = (patches%14)*delta;
  
  if(filter=="DoG")
    img = DoGFilter(image);
  else if(filter=="CSDN")
    img = CSDNFilter(image);
  else if(filter=="Gaussian")
    img = GaussianFilter(image);
  else
    cerr << "Error, no filter choosed" << endl;
  
  img = img(Rect(i,j,size,size));
  //img.convertTo(img, CV_32F); // Retirar isso depois
  //dct(img,img);			// Isso tambem
  
  if(descriptor=="SIFT")
    calcSIFTDescriptors(img,temp);
  else if(descriptor=="MLBP")
    calcLBPHistogram(img,temp);
  else
    cerr << "Error, no descriptor choosed" << endl;
  
  normalize(temp,temp,1);
  result = temp.t();
  
  return result;
} 

float Kernel::calcKernel(Mat x, Mat y){
  float result = 0;
  for (int i = 0; i < x.rows; i++)
    result += x.at<float>(i) * y.at<float>(i);
  
  //result = result/(norm(x,NORM_L2)*norm(y,NORM_L2));
  result = result*result*result;
  if(result!=result)
    result = 0;
    
  return result;
}
