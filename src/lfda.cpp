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

LFDA::LFDA(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches, int size, int overlap)
{
  this->trainingPhotos=trainingPhotos;
  this->trainingSketches=trainingSketches;
  this->size = size;
  this->overlap = overlap;
}

LFDA::~LFDA()
{
  
}

void LFDA::compute()
{
  for(int i=0; i < this->trainingSketches.size(); i++){
    vector<Mat> phi = this->extractDescriptors(this->trainingSketches[i],this->size,this->overlap);
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
    vector<Mat> phi = this->extractDescriptors(this->trainingPhotos[i],this->size,this->overlap);
    for(int j=0; j<phi.size(); j++){
      if(i==0)
	this->Xpk.push_back(phi[j]);
      else
	hconcat(this->Xpk[j], phi[j], this->Xpk[j]);
      
      hconcat(this->Xk[j], phi[j], this->Xk[j]);
    }
  }
  
  
  for(int i=0; i < Xk.size(); i++){
   
    Mat Xk_mean=Xk[i].col(0), Xpk_mean=Xpk[i].col(0), Xsk_mean=Xsk[i].col(0);
    
    int ncols = Xk[i].cols; 
    for(int j=0; j<ncols; j++)
	Xk_mean+=Xk[i].col(j);
    
    Xk_mean = Xk_mean*(1.0/ncols);
    
    for(int j=0; j<ncols; j++)
      Xk[i].col(j)-=Xk_mean;

    
    ncols = Xpk[i].cols; 
    for(int j=0; j<ncols; j++)
	Xpk_mean+=Xpk[i].col(j);
    
    Xpk_mean = Xpk_mean*(1.0/ncols);
    
    for(int j=0; j<ncols; j++)
      Xpk[i].col(j)-=Xpk_mean;
      
    
    ncols = Xsk[i].cols; 
    for(int j=0; j<ncols; j++)
	Xsk_mean+=Xsk[i].col(j);
    
    Xsk_mean = Xsk_mean*(1.0/ncols);
    
    for(int j=0; j<ncols; j++)
      Xsk[i].col(j)-=Xsk_mean;
    
    PCA pca(this->Xk[i],Mat(),CV_PCA_DATA_AS_COL,100);
    Mat Wk = pca.eigenvectors.t();
    
    Mat Yk = Wk.t()*(this->Xsk[i]+this->Xpk[i])*(0.5);
    
    Mat XXsk = Wk.t()*this->Xsk[i]-Yk;
    Mat XXpk = Wk.t()*this->Xpk[i]-Yk;
    
    Mat XXk;
    hconcat(XXsk,XXpk,XXk);
    pca(XXk,Mat(),CV_PCA_DATA_AS_COL,100);
    
    Mat VVk = pca.eigenvectors.t();
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
    
    Mat Vk = (valsDiag*VVk.t()).t();
    
    pca(Vk.t()*Yk,Mat(),CV_PCA_DATA_AS_COL,99);
    Mat Uk = pca.eigenvectors.t(); // Deveria ser (100x99)
    
  /*  
    LDA lda;
    Mat temp = Wk.t()*Xk[i];
    temp = temp.t();
    vector<int> _classes;
    for(int i=0; i<ncols; i++){
      _classes.push_back(i);
      _classes.push_back(i);
    }
    lda.compute(temp, _classes);
    Mat temp2 = lda.eigenvectors();
    temp2.convertTo(temp2, CV_32F);
    Mat omega = (temp2*Wk.t()).t();
  */
  
    Mat omega = Wk*Vk*Uk; // Deveria ser (99x10920)
    
    /* cout << this->Xk.size() << endl <<
     *    Xk[i].size() << endl <<
     *    Yk.size() << endl <<
     *    Wk.size() << endl <<
     *    XXsk.size() << endl <<
     *    XXk.size() << endl <<
     *    VVk.size() << endl <<
     *    valsDiag.size() << endl <<
     *    Vk.size() << endl <<
     *    Uk.size() << endl <<
     *    omega.size() << endl;
     */
    this->omegaK.push_back(omega);
  }
}

Mat LFDA::project(Mat image)
{
  vector<Mat> phi = this->extractDescriptors(image,this->size,this->overlap);
  Mat result = this->omegaK[0].t()*phi[0];
  for(int j=1; j<phi.size(); j++){
    vconcat(result, this->omegaK[j].t()*phi[j], result);
  }
  
  normalize(result, result, 1);
  
  return result;
}

vector<Mat> LFDA::extractDescriptors(Mat img, int size, int delta){
  int w = img.cols, h=img.rows;
  vector<Mat> result;
  
  for(int i=0;i<=w-size;i+=(size-delta)){
    Mat aux, temp;
    for(int j=0;j<=h-size;j+=(size-delta)){
      Mat a, b;
      calcSIFTDescriptors(img(Rect(i,j,size,size)),a);
      //normalize(a,a,1);
      calcLBPHistogram(img(Rect(i,j,size,size)),b);
      //normalize(b,b,1);
      hconcat(a,b,temp);
      if(aux.empty())
	aux = temp.clone();
      else
	hconcat(aux, temp, aux);
    }
    result.push_back(aux.t());
  }
  
  return result;
}
