/*
 * utils.cpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#include "utils.hpp"

void loadImages(string src, vector<Mat> &dest, float proportion){
  directory_iterator end;
  
  path dir(src);
  
  string filename;
  int n=0;
  int num;
  Mat temp;
  
  for (directory_iterator pos(dir); pos != end; ++pos){
    if(is_regular_file(*pos)){
      n++;
    }
  }
  
  dest.resize(n);
  
  for (directory_iterator pos(dir); pos != end; ++pos){
    if(is_regular_file(*pos)){
      filename = pos->path().filename().string();
      num = atoi((filename.substr(0,filename.find("."))).c_str());
      temp = imread(pos->path().c_str(), 0);
      resize(temp, temp, Size(temp.size().width*proportion,
			      temp.size().height*proportion));
      dest[num-1] = Mat(temp(Rect(temp.cols/2-100,
				  temp.rows/2-125, 200, 250)));
      // elbp(temp,1,8);
      //dest[num-1].convertTo(dest[num-1],CV_8U);
    }
  }
  
}


void createFolds(vector<Mat>& input, vector<vector<Mat> >&output, int num){
  vector<Mat>::iterator it;
  int size = (input.size()/num)*num;
  
  for(int i=0; i<num; i++){
    output.push_back(vector<Mat>());
  }
  
  for(int i = 0; i < size; i++) {
    output[i%num].push_back(input[i]);
  }
}

float euclideanDistance(Mat a, Mat b){
  float result = 0;
  Mat temp;
  pow(a-b,2,temp);
  result = sum(temp).val[0];
  return result;
}

void patcher(Mat img, int size, int delta, vector<vector<Mat> > &result){
  int w = img.cols, h=img.rows;
  
  for(int i=0;i<=h-size;i+=(size-delta)){
    vector<Mat> col;
    for(int j=0;j<=w-size;j+=(size-delta)){
      col.push_back(img(Rect(j,i,size,size)));
    }
    result.push_back(col);
  }
  
}

void myPCA(vector<Mat> &trainingSet, Mat &vecs, Mat &valsDiag){
  
  Size imSize = trainingSet[0].size();
  Mat avg = Mat::zeros(imSize, CV_32F);
  Mat temp = Mat::zeros(imSize, CV_32F);
  Mat p;
  
  int n = trainingSet.size();
  
  for(int i=0; i<n ; i++){
    trainingSet[i].convertTo(temp, CV_32F);
    avg += temp;
  }
  
  avg = avg/n;
  p = Mat::zeros(imSize.area(), n, CV_32F);
  
  for(int i=0; i<n; i++){
    trainingSet[i].convertTo(temp, CV_32F);
    p.col(i) = temp.reshape(1, imSize.area()) - avg.reshape(1, imSize.area());
  }
  
  Mat ppt = p*(p.t());
  Mat vals;
  MatrixXf e_M;
  
  cv2eigen(ppt, e_M);
  
  EigenSolver<MatrixXf> es(e_M);
  
  eigen2cv(MatrixXf(es.eigenvectors().real()), vecs);
  eigen2cv(MatrixXf(es.eigenvalues().real()), vals);
  
  valsDiag = Mat::zeros(vecs.size(), CV_32F);
  
  float aux;
  
  for(int i=0; i<n; i++){
    aux = vals.at<float>(i);//1/sqrt(vals.at<float>(i));
    if(aux!=aux)
      valsDiag.at<float>(i,i) = 0;
    else
      valsDiag.at<float>(i,i) = aux;
  }  
}