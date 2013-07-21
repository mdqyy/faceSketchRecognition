/*
 * eigentransformation.cpp
 *
 *  Created on: 29/01/2013
 *      Author: marco
 */

#include "eigenTransformation.hpp"

void createEigenSpace(vector<Mat> &trainingSet, Mat &p, Mat &avg, Mat &vecs, Mat &valsDiag, Mat &eigenSpace){

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
