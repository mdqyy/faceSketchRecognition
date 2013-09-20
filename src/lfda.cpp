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


#include "lfda.hpp"

LFDA::LFDA(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches)
{
  this->trainingPhotos=trainingPhotos;
  this->trainingSketches=trainingSketches;
  
  //this->nTraining=this->trainingPhotos.size();
}

LFDA::~LFDA()
{
  
}

void LFDA::compute()
{
  for(int i=0; i < this->trainingSketches.size(); i++){
    vector<Mat> phi = this->extractDescriptors(this->trainingSketches[i],16,8);
    for(int j=0; j<phi.size(); j++){
      if(i==0){
	this->Xsk.push_back(phi[j]);
	this->Xk.push_back(phi[j]);
      }
      else{
	hconcat(this->Xsk[j], phi[j], this->Xsk[j]);
	hconcat(this->Xk[j], phi[j], this->Xk[j]);
      }
    }
  }
  
  for(int i=0; i < this->trainingPhotos.size(); i++){
    vector<Mat> phi = this->extractDescriptors(this->trainingPhotos[i],16,8);
    for(int j=0; j<phi.size(); j++){
      if(i==0)
	this->Xpk.push_back(phi[j]);
      else
	hconcat(this->Xpk[j], phi[j], this->Xpk[j]);
	
      hconcat(this->Xk[j], phi[j], this->Xk[j]);
    }
  }
  
  for(int i=0; i < Xk.size(); i++){
    
    PCA pca(this->Xk[i],Mat(),CV_PCA_DATA_AS_COL,100);
    Mat Wk = pca.eigenvectors.clone();
    
    Mat Yk = Wk*(this->Xsk[i]+this->Xpk[i])*(0.5);
    
    Mat XXsk = Wk*this->Xsk[i]-Yk;
    Mat XXpk = Wk*this->Xpk[i]-Yk;
    
    Mat XXk;
    hconcat(XXsk,XXpk,XXk);
    pca(XXk,Mat(),CV_PCA_DATA_AS_COL,100);
    
    Mat VVk = pca.eigenvectors.clone();
    Mat Diagk = pca.eigenvalues.clone();
    
    Mat valsDiag = Mat::zeros(VVk.size(), CV_32F);
    
    float aux;
    
    for(int i=0; i<valsDiag.cols; i++){
      aux = 1/sqrt(Diagk.at<float>(i));
      if(aux!=aux)
	valsDiag.at<float>(i,i) = 0;
      else
	valsDiag.at<float>(i,i) = aux;
    }
    
    Mat Vk = (valsDiag*VVk).t();
    
    pca(Vk.t()*Yk,Mat(),CV_PCA_DATA_AS_COL,99);
    Mat Uk = pca.eigenvectors.clone(); // Deveria ser (100x99)
    
    Mat omega = Wk.t()*Vk*Uk.t(); // Deveria ser (99x10920)
    
   /* cout << this->Xk.size() << endl <<
    Xk[i].size() << endl <<
    Yk.size() << endl <<
    Wk.size() << endl <<
    XXsk.size() << endl <<
    XXk.size() << endl <<
    VVk.size() << endl <<
    valsDiag.size() << endl <<
    Vk.size() << endl <<
    Uk.size() << endl <<
    omega.size() << endl;
    */
    this->omegaK.push_back(omega);
  }
}

Mat LFDA::project(Mat& image)
{
  vector<Mat> phi = this->extractDescriptors(image,16,8);
  Mat result = this->omegaK[0].t()*phi[0];
    for(int j=1; j<phi.size(); j++){
      vconcat(result, this->omegaK[j].t()*phi[j], result);
    }
    
    normalize(result, result, 1);
    
  return result;
}

vector<Mat> LFDA::extractDescriptors(Mat& img, int size, int delta){
  int w = img.cols, h=img.rows;
  vector<Mat> result;
  
  for(int i=0;i<=w-size;i+=(size-delta)){
    Mat aux, temp;
    for(int j=0;j<=h-size;j+=(size-delta)){
      Mat a, b;
      calcSIFTDescriptors(img(Rect(i,j,size,size)),a);
      calcLBPHistogram(img(Rect(i,j,size,size)),b);
      hconcat(a,b,temp);
      if(aux.empty())
	aux = temp.clone();
      else
	hconcat(aux, temp, aux);
    }
    result.push_back(aux.clone().t());
  }
  
  return result;
}
