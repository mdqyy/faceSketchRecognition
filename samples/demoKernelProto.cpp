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
  testingPhotos,testingSketches;
  
  vector<Mat> photos, sketches, extra, vphotos, vsketches;
  
  loadImages(argv[1],vphotos,1);
  loadImages(argv[2],vsketches,1);
  loadImages(argv[3],photos,1);
  loadImages(argv[4],sketches,1);
  //loadImages(argv[5],extra,1);
  
  
  testingPhotos.insert(testingPhotos.end(),photos.begin(),photos.end());
  testingSketches.insert(testingSketches.end(),sketches.begin(),sketches.end());
  testingPhotos.insert(testingPhotos.end(), extra.begin(), extra.end());
  
  trainingPhotos.insert(trainingPhotos.end(),vphotos.begin(),vphotos.begin()+10);
  trainingSketches.insert(trainingSketches.end(),vsketches.begin(),vsketches.begin()+10);
  
  vphotos.clear();
  vsketches.clear();
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
  
  cout << k.projectGallery(testingPhotos[53]) << endl;
  cout << k.projectProbe(testingSketches[53]) << endl;
  cout << endl;
  cout << k.projectGallery(testingPhotos[1]) << endl;
  cout << k.projectProbe(testingSketches[1]) << endl;
  cout << endl;
  cout << k.projectGallery(testingPhotos[100]) << endl;
  cout << k.projectProbe(testingSketches[100]) << endl;
  return 0;
}
