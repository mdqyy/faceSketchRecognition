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


#ifndef EIGENTRANSFORMATION_HPP
#define EIGENTRANSFORMATION_HPP

#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include <vector>

using namespace cv;
using namespace Eigen;

class Eigentransformation
{

public:
Eigentransformation();
Eigentransformation(const Eigentransformation& other);
virtual ~Eigentransformation();
virtual Eigentransformation& operator=(const Eigentransformation& other);
virtual bool operator==(const Eigentransformation& other) const;
static void createEigenSpace(vector<Mat>&, Mat&, Mat&, Mat&, Mat&, Mat&);
};

#endif // EIGENTRANSFORMATION_HPP
