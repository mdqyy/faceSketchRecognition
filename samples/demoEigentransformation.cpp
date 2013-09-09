/*
 * Test.cpp
 *
 *  Created on: 08/12/2012
 *      Author: marco
 */

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
  
  //loadImages(argv[3],extra,1);
  loadImages(argv[1],vphotos,1);
  loadImages(argv[2],vsketches,1);
  loadImages(argv[3],photos,1);
  loadImages(argv[4],sketches,1);
  
  
  testingPhotos.insert(testingPhotos.end(),vphotos.begin()+88,vphotos.end());
  testingSketches.insert(testingSketches.end(),vsketches.begin()+88,vsketches.end());
  
  trainingPhotos.insert(trainingPhotos.end(),vphotos.begin(),vphotos.begin()+88);
  trainingSketches.insert(trainingSketches.end(),vsketches.begin(),vsketches.begin()+88);
  
  //testingPhotos.insert(testingPhotos.end(), extra.begin(), extra.begin()+999);
  
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
  for(int i=0; i<nTestingSketches; i++){
    eigenT.projectSketch(testingSketches[i],sketchesB[i],sketchesContr[i],recPhotosB[i]);
  }
  
  cerr << "projecting testing photos" << endl;
  for(int i=0; i<nTestingPhotos; i++){  
    eigenT.projectPhoto(testingPhotos[i],photosB[i],photosContr[i],recSketchesB[i]);
  }
  
  vector<int> d1(nTestingSketches), d2(nTestingSketches), d3(nTestingSketches);
  
  cerr << "calc d1" << endl;
  for(int i=0; i<nTestingSketches; i++){
    float val = euclideanDistance(photosContr[i],sketchesContr[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(euclideanDistance(photosContr[j],sketchesContr[i])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d1= "<< val << endl;
	temp++;
      }
    }
    d1[i] = temp;
  }
  
  cerr << "calc d2" << endl;
  for(int i=0; i<nTestingSketches; i++){
    float val = euclideanDistance(recSketchesB[i],sketchesB[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d2= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(euclideanDistance(recSketchesB[j],sketchesB[i])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d2= "<< val << endl;
	temp++;
      }
    }
    d2[i] = temp;
  }
  
  cerr << "calc d3" << endl;
  for(int i=0; i<nTestingSketches; i++){
    float val = euclideanDistance(recPhotosB[i],photosB[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d3= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(euclideanDistance(recPhotosB[i],photosB[j])<= val && i!=j){
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