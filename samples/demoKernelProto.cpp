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
  testingPhotos,testingSketches, svmTrainingPhotos,svmTrainingSketches, photos, sketches, extra;
  
  loadImages(argv[1],photos,1);
  loadImages(argv[2],sketches,1);
  //loadImages(argv[5],extra,1);
  
  if(photos.size()!=sketches.size())
    return -1;
  
  for(int i=383; i<photos.size()-3; i+=4){
    trainingPhotos.push_back(photos[i]);
    trainingPhotos.push_back(photos[i+1]);
    testingPhotos.push_back(photos[i+2]);
    trainingSketches.push_back(sketches[i]);
    trainingSketches.push_back(sketches[i+1]);
    testingSketches.push_back(sketches[i+2]);

    svmTrainingPhotos.push_back(photos[i+3]);
    svmTrainingSketches.push_back(sketches[i+3]);
  }
  
  
  testingPhotos.insert(testingPhotos.end(),extra.begin(),extra.end());
  
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size(),
  nSVMTraining = svmTrainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  vector<Mat> testingPhotosFinal(nTestingPhotos), testingSketchesFinal(nTestingSketches), 
  svmTrainingPhotosFinal(nSVMTraining), svmTrainingSketchesFinal(nSVMTraining);
  
  for(int patch=0; patch<154; patch++){
    
    Kernel k(trainingPhotos, trainingSketches, patch, "CSDN", "SIFT");
    k.compute();
    
    //cerr << "calculating patch " << patch <<  endl;
    
    for(int i=0; i<nTestingPhotos; i++){
      Mat temp = k.projectGallery(testingPhotos[i]).clone();
      normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
      if(patch==0)
	testingPhotosFinal[i] = temp;
      else
	vconcat(testingPhotosFinal[i], temp, testingPhotosFinal[i]);
    }
    
    for(int i=0; i<nTestingSketches; i++){
      Mat temp = k.projectProbe(testingSketches[i]).clone();
      normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
      if(patch==0)
	testingSketchesFinal[i] = temp;
      else
	vconcat(testingSketchesFinal[i], temp, testingSketchesFinal[i]);
    }
    
    for(int i=0; i<nSVMTraining; i++){
      Mat temp = k.projectGallery(svmTrainingPhotos[i]).clone();
      normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
      if(patch==0)
	svmTrainingPhotosFinal[i] = temp;
      else
	vconcat(svmTrainingPhotosFinal[i], temp, svmTrainingPhotosFinal[i]);
    }
    
    for(int i=0; i<nSVMTraining; i++){
      Mat temp = k.projectProbe(svmTrainingSketches[i]).clone();
      normalize(temp,temp,1,0,NORM_MINMAX, CV_32F);
      if(patch==0)
	svmTrainingSketchesFinal[i] = temp;
      else
	vconcat(svmTrainingSketchesFinal[i], temp, svmTrainingSketchesFinal[i]);
    }
  }
  
  // Set up training data
  Mat labelsMat;
    
  Mat trainingDataMat = svmTrainingPhotosFinal[0]-svmTrainingSketchesFinal[0];
  labelsMat.push_back<float>(1);
  
  for(int i=1; i<nSVMTraining; i++){
    hconcat(trainingDataMat,Mat(svmTrainingPhotosFinal[i]-svmTrainingSketchesFinal[i]),trainingDataMat);
    labelsMat.push_back<float>(1);
  }
  
  for(int i=0; i<nSVMTraining; i++){
    for(int j=0; j<nSVMTraining; j++){
      if(i!=j){
	hconcat(trainingDataMat,Mat(svmTrainingPhotosFinal[i]-svmTrainingSketchesFinal[j]),trainingDataMat);
	labelsMat.push_back<float>(-1);
      }
    }
  }
  
  trainingDataMat = trainingDataMat.t();

  // Set up SVM's parameters
  CvSVMParams params;
  
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
  params.degree = 0; // for poly
  params.gamma = 20; // for poly/rbf/sigmoid
  params.coef0 = 0; // for poly/sigmoid
  
  params.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
  params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
  params.p = 0.0; // for CV_SVM_EPS_SVR
  
  params.class_weights = NULL; // for CV_SVM_C_SVC
  params.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
  params.term_crit.max_iter = 1000;
  params.term_crit.epsilon = 1e-6;  
  
  // Train the SVM
  CvSVM SVM;
  SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params);
  
  
  vector<int> rankSVM(nTestingSketches);
  
  for(int i=0; i<nTestingSketches; i++){
    float val = SVM.predict(Mat(testingPhotosFinal[i]-testingSketchesFinal[i]), true);//norm(testingPhotosFinal[i],testingSketchesFinal[i], NORM_L2);
    //cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(SVM.predict(Mat(testingPhotosFinal[j]-testingSketchesFinal[i]), true)<= val && i!=j){//(norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2)<= val && i!=j){
	//cerr << "small "<< j << " d1= "<< SVM.predict(Mat(testingPhotosFinal[j]-testingSketchesFinal[i]), true) << endl;//norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2) << endl;
	temp++;
      }
    }
    rankSVM[i] = temp;
    cerr << i << " rank= " << temp << endl;
  }
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
    cerr << "Rank "<< i << ": ";
    cerr << "d1= " << (float)count_if(rankSVM.begin(), rankSVM.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }
 
  vector<int> rank(nTestingSketches);
  
  for(int i=0; i<nTestingSketches; i++){
    float val = norm(testingPhotosFinal[i],testingSketchesFinal[i], NORM_L2);
    //cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2)<= val && i!=j){
	//cerr << "small "<< j << " d1= "<< norm(testingPhotosFinal[j],testingSketchesFinal[i], NORM_L2) << endl;
	temp++;
      }
    }
    rank[i] = temp;
    cerr << i << " rank= " << temp << endl;
  }
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
    cerr << "Rank "<< i << ": ";
    cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }

  
  return 0;
}
