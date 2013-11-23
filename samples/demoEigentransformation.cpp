#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "eigentransformation.hpp"
#include "utils.hpp"


using namespace cv;
using namespace std;

int main(int argc, char** argv){
  
  /*	if(argc != 5){
   *		cout << "Usage: "<< argv[0] << "<photos> <sketches>"
   *				<< endl;
   *		return -1;
}
*/
    
  vector<Mat> trainingPhotos,	trainingSketches,
  testingPhotos,testingSketches;
  
  vector<Mat> photos, sketches, extra, vphotos, vsketches;
  
  loadImages(argv[1],photos,1);
  loadImages(argv[2],sketches,1);
  //loadImages(argv[5],extra,1);
  
  if(photos.size()!=sketches.size())
    return -1;
  
  for(int i=383; i<photos.size()-2; i+=3){
    trainingPhotos.push_back(photos[i]);
    trainingPhotos.push_back(photos[i+1]);
    testingPhotos.push_back(photos[i+2]);
    trainingSketches.push_back(sketches[i]);
    trainingSketches.push_back(sketches[i+1]);
    testingSketches.push_back(sketches[i+2]);
  }
  
  testingPhotos.insert(testingPhotos.end(),extra.begin(),extra.end());
  
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  Eigentransformation eigenT(trainingPhotos,trainingSketches);
  eigenT.compute();
  
  vector<Mat> photosB(nTestingPhotos), sketchesB(nTestingSketches),
  photosContr(nTestingPhotos), sketchesContr(nTestingSketches),
  recPhotosB(nTestingSketches), recSketchesB(nTestingPhotos);
  
  Mat image;
  
  cerr << "projecting testing sketches" << endl;
  
  #pragma omp parallel for
  for(int i=0; i<nTestingSketches; i++){
    eigenT.projectSketch(testingSketches[i],sketchesB[i],sketchesContr[i],recPhotosB[i]);
  }
  
  cerr << "projecting testing photos" << endl;
  #pragma omp parallel for
  for(int i=0; i<nTestingPhotos; i++){  
    eigenT.projectPhoto(testingPhotos[i],photosB[i],photosContr[i],recSketchesB[i]);
  }
  
  vector<int> d1(nTestingSketches), d2(nTestingSketches), d3(nTestingSketches);
  
  cerr << "calc d1" << endl;
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(photosContr[i],sketchesContr[i], NORM_L2);//euclideanDistance(photosContr[i],sketchesContr[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(photosContr[j],sketchesContr[i],NORM_L2)<= val && i!=j){//euclideanDistance(photosContr[j],sketchesContr[i])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d1= "<< val << endl;
	temp++;
      }
    }
    d1[i] = temp;
  }
  
  cerr << "calc d2" << endl;
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(recSketchesB[i],sketchesB[i],NORM_L2);//euclideanDistance(recSketchesB[i],sketchesB[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d2= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(recSketchesB[j],sketchesB[i],NORM_L2)<= val && i!=j){//euclideanDistance(recSketchesB[j],sketchesB[i])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d2= "<< val << endl;
	temp++;
      }
    }
    d2[i] = temp;
  }
  
  cerr << "calc d3" << endl;
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(recPhotosB[i],photosB[i],NORM_L2);//euclideanDistance(recPhotosB[i],photosB[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d3= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(recPhotosB[i],photosB[j],NORM_L2)<= val && i!=j){//euclideanDistance(recPhotosB[i],photosB[j])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d3= "<< val << endl;
	temp++;
      }
    }
    d3[i] = temp;
  }
  
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
  cerr << "Rank "<< i << ": ";
  cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), [i](int x) {return x < i;})/nTestingSketches;
  cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), [i](int x) {return x < i;})/nTestingSketches;
  cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }
    
  return 0;
}
