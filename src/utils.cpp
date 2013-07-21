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
