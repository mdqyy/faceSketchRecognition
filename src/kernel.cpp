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

Kernel::Kernel(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches)
{
  this->trainingPhotos=trainingPhotos;
  this->trainingSketches=trainingSketches;
  
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
  
  Kp = Mat::zeros(n/2,n/2,CV_64F);
  Kg = Mat::zeros(n/2,n/2,CV_64F);
  
  for(int i=0; i<n/2; i++)
    for(int j=0; j<n/2; j++){
      Kg.at<double>(i,j) = this->cosineKernel(trainingPhotosDescriptors[i], trainingPhotosDescriptors[j]);
      Kp.at<double>(i,j) = this->cosineKernel(trainingSketchesDescriptors[i],trainingSketchesDescriptors[j]);
    }
    
  R = Kg*((Kp).t()*Kp).inv()*(Kp).t();
  
  R.convertTo(R, CV_32F);
  
  vector<int> _classes;
  
  T2.push_back(this->projectProbeIntern(trainingSketchesDescriptors[0]));
  hconcat(T2,projectGalleryIntern(trainingPhotosDescriptors[0]),T2);
  
  for(int i=n/2+1; i<n; i++){
    hconcat(T2,projectProbeIntern(trainingSketchesDescriptors[i]),T2);
    hconcat(T2,projectGalleryIntern(trainingPhotosDescriptors[i]),T2);
  }
  
  this->pca.computeVar(T2, Mat(), CV_PCA_DATA_AS_COL, 0.99);
  
  this->mean = pca.mean.clone();
  cout << mean.size() << endl;
  cout << pca.eigenvectors.size() << endl;
  cout << T2.size() << endl;
  
  Mat T2_pca = pca.eigenvectors*T2;
  T2_pca = T2_pca.t();
  
  cout << T2_pca.size() << endl; 
   
  for(int i=0; i<n/2; i++){  
    _classes.push_back(i);
    _classes.push_back(i);
  }
  
  lda.compute(T2_pca, _classes);
  cout << lda.eigenvectors().size() << endl;
}

Mat Kernel::projectGalleryIntern(Mat desc)
{
  int n = trainingPhotosDescriptors.size();
  Mat result = Mat::zeros(1,n/2,CV_32F);
  for(int i=0; i<n/2; i++)
    result.at<float>(i) = this->cosineKernel(desc,trainingPhotosDescriptors[i]);
   
  return result.t();
}

Mat Kernel::projectProbeIntern(Mat desc)
{
  int n = trainingSketchesDescriptors.size();
  Mat result = Mat::zeros(1,n/2,CV_32F);
  for(int i=0; i<n/2; i++)
    result.at<float>(i) = this->cosineKernel(desc,trainingSketchesDescriptors[i]);
  
  return R*result.t();
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
  //img = DoGFilter(image);
  //img = CSDNFilter(image);
  img = GaussianFilter(image);
  
  for(int i=0;i<=w-size;i+=(size-delta)){
    for(int j=0;j<=h-size;j+=(size-delta)){
      //calcSIFTDescriptors(img(Rect(i,j,size,size)),temp);
      calcLBPHistogram(img(Rect(i,j,size,size)),temp);
      normalize(temp,temp,1);
      if(result.empty())
	result = temp.t();
      else
	vconcat(result, temp.t(), result);
    }
  }
  return result;
} 

double Kernel::cosineKernel(Mat x, Mat y){
  double result = 0;
  for (int i = 0; i < x.rows; i++)
    result += x.at<float>(i) * y.at<float>(i);
  
  result = result/(norm(x,NORM_L2)*norm(y,NORM_L2));
  //result = result*result*result;
  if(result!=result)
    cout << result << endl;
  
  return result;
}
