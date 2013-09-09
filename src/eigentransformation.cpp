/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2013  <copyright holder> <email>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "eigentransformation.hpp"

Eigentransformation::Eigentransformation(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches)
{
  this->trainingPhotos=trainingPhotos;
  this->trainingSketches=trainingSketches;

  this->nTraining=this->trainingPhotos.size();
}

Eigentransformation::~Eigentransformation()
{

}

void Eigentransformation::createEigenSpace(vector<Mat> &trainingSet, Mat &p, Mat &avg, Mat &vecs, Mat &valsDiag, Mat &eigenSpace){

	Size imSize = trainingSet[0].size();
	avg = Mat::zeros(imSize, CV_32F);
	Mat temp = Mat::zeros(imSize, CV_32F);

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

	Mat ptp = (p.t())*p;
	Mat vals;
	MatrixXf e_M;

	cv2eigen(ptp, e_M);

	EigenSolver<MatrixXf> es(e_M);

	eigen2cv(MatrixXf(es.eigenvectors().real()), vecs);
	eigen2cv(MatrixXf(es.eigenvalues().real()), vals);

	valsDiag = Mat::zeros(vecs.size(), CV_32F);

	float aux;

	for(int i=0; i<n; i++){
		aux = 1/sqrt(vals.at<float>(i));
		if(aux!=aux)
			valsDiag.at<float>(i,i) = 0;
		else
			valsDiag.at<float>(i,i) = aux;
	}

	eigenSpace = p*vecs*valsDiag;
}

void Eigentransformation::compute()
{
  this->createEigenSpace(this->trainingPhotos,this->pPhotos,this->avgPhoto,this->vecsPhotos,
			 this->valsDiagPhotos,this->eigenPhotos);
  this->createEigenSpace(this->trainingSketches,this->pSketches,this->avgSketch,this->vecsSketches,
			 this->valsDiagSketches,this->eigenSketches);
}

void Eigentransformation::projectPhoto(Mat& img, Mat& photoB, Mat& photoContr, Mat& recSketchB)
{		
		Mat image;
		image = img.clone();
		image.convertTo(image, CV_32F);
		image -= this->avgPhoto;
		image = image.reshape(1, this->eigenPhotos.rows);
		photoB = this->eigenPhotos.t()*image;
		photoContr = this->vecsPhotos*this->valsDiagPhotos*photoB;
		photoContr -= ((float)sum(photoContr)[0] - 1)/this->nTraining;
		recSketchB = this->eigenSketches.t()*this->pSketches*photoContr;
}

void Eigentransformation::projectSketch(Mat& img, Mat& sketchB, Mat& sketchContr, Mat& recPhotoB)
{
		Mat image;
		image = img.clone();
		image.convertTo(image, CV_32F);
		image -= this->avgSketch;
		image = image.reshape(1, this->eigenSketches.rows);
		sketchB = this->eigenSketches.t()*image;
		sketchContr = this->vecsSketches*this->valsDiagSketches*sketchB;
		sketchContr -= ((float)sum(sketchContr)[0] - 1)/this->nTraining;
		recPhotoB = this->eigenPhotos.t()*this->pPhotos*sketchContr;
}