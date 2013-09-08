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

Eigentransformation::Eigentransformation()
{

}

Eigentransformation::Eigentransformation(const Eigentransformation& other)
{

}

Eigentransformation::~Eigentransformation()
{

}

Eigentransformation& Eigentransformation::operator=(const Eigentransformation& other)
{
return *this;
}

bool Eigentransformation::operator==(const Eigentransformation& other) const
{
///TODO: return ...;
return false;
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

