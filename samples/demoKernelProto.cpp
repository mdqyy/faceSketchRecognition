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
  
  trainingPhotos.insert(trainingPhotos.end(),vphotos.begin(),vphotos.begin()+300);
  trainingSketches.insert(trainingSketches.end(),vsketches.begin(),vsketches.begin()+300);
  
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
 
  
  vector<int> rank(nTestingSketches);
  
  cerr << "calculating distances" << endl;
  
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(k.projectGallery(testingPhotos[i]),k.projectProbe(testingSketches[i]), NORM_L2);
    cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(k.projectGallery(testingPhotos[j]),k.projectProbe(testingSketches[i]), NORM_L2)<= val && i!=j){
	cerr << "small "<< j << " d1= "<< norm(k.projectGallery(testingPhotos[j]),k.projectProbe(testingSketches[i]), NORM_L2) << endl;
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
