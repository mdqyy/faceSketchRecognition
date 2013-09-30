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
  
  for(int i=0; i<photos.size()-2; i+=3){
    trainingPhotos.push_back(photos[i]);
    trainingPhotos.push_back(photos[i+1]);
    testingPhotos.push_back(photos[i+2]);
    trainingSketches.push_back(sketches[i]);
    trainingSketches.push_back(sketches[i+1]);
    testingSketches.push_back(sketches[i+2]);
  }
  
  photos.clear();
  sketches.clear();
  extra.clear();
    
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  Kernel k(trainingPhotos, trainingSketches);
  
  k.compute();
 
  
  vector<int> rank(nTestingSketches);
  
  cerr << "calculating distances" << endl;
  
  vector<Mat> testingPhotosFinal(nTestingPhotos), testingSketchesFinal(nTestingSketches);
  
  for(int i=0; i<nTestingPhotos; i++)
    testingPhotosFinal[i] = k.projectGallery(testingPhotos[i]);
  
  for(int i=0; i<nTestingSketches; i++)
    testingSketchesFinal[i] = k.projectProbe(testingSketches[i]);
  
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(testingPhotosFinal[i],testingSketchesFinal[i], NORM_L2);
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
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
    cerr << "Rank "<< i << ": ";
    cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }


  return 0;
}
