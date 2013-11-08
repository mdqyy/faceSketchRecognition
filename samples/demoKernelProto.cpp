#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "descriptors.hpp"
#include "utils.hpp"
#include "kernel.hpp"
#include "filters.hpp"

#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  
  vector<Mat> trainingPhotos,	trainingSketches,
  testingPhotos,testingSketches, photos, sketches, extra;
    
  loadImages(argv[1],photos,1);
  loadImages(argv[2],sketches,1);
  //loadImages(argv[5],extra,1);
  
  if(photos.size()!=sketches.size())
    return -1;
  
  for(int i=0; i<photos.size()-1; i+=2){
    trainingPhotos.push_back(photos[i]);
    testingPhotos.push_back(photos[i+1]);
    trainingSketches.push_back(sketches[i]);
    testingSketches.push_back(sketches[i+1]);
  }
  
  testingPhotos.insert(testingPhotos.end(),extra.begin(),extra.end());
  
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  vector<Mat> testingPhotosFinal(nTestingPhotos), testingSketchesFinal(nTestingSketches);
  
  for(string filter : {"Gaussian", "DoG", "CSDN"}){
    for(string desc : {"SIFT", "MLBP"}){
      for(int patch=0; patch<154; patch++){
	
	Kernel k(trainingPhotos, trainingSketches, patch, filter, desc);
	k.compute();
	//cerr << "calculating patch " << patch <<  endl;
	
	for(int i=0; i<nTestingPhotos; i++){
	  Mat temp = k.projectGallery(testingPhotos[i]).clone();
	  normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
	  if(testingPhotosFinal[i].empty())
	    testingPhotosFinal[i] = temp;
	  else
	    vconcat(testingPhotosFinal[i], temp, testingPhotosFinal[i]);
	}
	
	for(int i=0; i<nTestingSketches; i++){
	  Mat temp = k.projectProbe(testingSketches[i]).clone();
	  normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
	  if(testingSketchesFinal[i].empty())
	    testingSketchesFinal[i] = temp;
	  else
	    vconcat(testingSketchesFinal[i], temp, testingSketchesFinal[i]);
	}
      }  
    }
  }
  
  vector<int> rank(nTestingSketches);
  
  for(int i=0; i<nTestingSketches; i++){
    float val = norm(testingPhotosFinal[i],testingSketchesFinal[i], NORM_L2);
    cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2)<= val && i!=j){
	cerr << "small "<< j << " d1= "<< norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2) << endl;
	temp++;
      }
    }
    rank[i] = temp;
    cerr << i << " rank= " << temp << endl;
  }
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100})
  {
    cerr << "Rank "<< i << ": ";
    cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }
  
  cerr << "------------------------------------------------------" << endl;
   
  // Set up training data
  Mat labelsMat;
  Mat svmTrainingtemp;
  svmTrainingtemp =  Mat::zeros(1,1,CV_32F); //Editar essa parte
  normalize(svmTrainingtemp,svmTrainingtemp,1,0,NORM_MINMAX, CV_32F);
  Mat trainingDataMat = svmTrainingtemp;
  labelsMat.push_back<float>(1);
  
  for(int i=1; i<100; i++){
    svmTrainingtemp =  Mat::zeros(1,1,CV_32F); //Editar essa parte
    normalize(svmTrainingtemp,svmTrainingtemp,1,0,NORM_MINMAX, CV_32F);
    hconcat(trainingDataMat,svmTrainingtemp,trainingDataMat);
    labelsMat.push_back<float>(1);
  }
  
  for(int i=0; i<100; i++){
    for(int j=0; j<100; j++){
      if(i!=j){
	svmTrainingtemp =  Mat::zeros(1,1,CV_32F); //Editar essa parte
	normalize(svmTrainingtemp,svmTrainingtemp,1,0,NORM_MINMAX, CV_32F);
	hconcat(trainingDataMat,svmTrainingtemp,trainingDataMat);
	labelsMat.push_back<float>(-1);
      }
    }
  }
  
  return 0;
}
