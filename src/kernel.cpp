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
  
}

Mat Kernel::project(Mat& image)
{
  
  return Mat();
}

double Kernel::cosineKernel(Mat& x, Mat& y)
{
  double result = 0;
  for (int i = 0; i < x.rows; i++)
    result += x.at<float>(i) * y.at<float>(i);
  
  result = result/(norm(x,NORM_L2)*norm(y,NORM_L2));
  return result;
}
