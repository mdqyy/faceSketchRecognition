#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "descriptors.hpp"
#include "utils.hpp"
#include "lfda.hpp"

#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  
  clock_t start,end;
  int tempo;
  
  start=clock();
  
  /*	if(argc != 5){
   *		cout << "Usage: "<< argv[0] << "<photos> <sketches>"
   *				<< endl;
   *		return -1;
}
*/
  
  vector<Mat> trainingPhotos,	trainingSketches,
  testingPhotos,testingSketches;
  
  vector<Mat> photos, sketches, extra, vphotos, vsketches;
  
  loadImages(argv[1],vphotos,1);
  loadImages(argv[2],vsketches,1);
  //loadImages(argv[3],photos,1);
  //loadImages(argv[4],sketches,1);
  //loadImages(argv[5],extra,1);
  
  if(vphotos.size()!=vsketches.size())
    return -1;
  
  for(int i=0; i<vphotos.size()-1; i+=2){
    trainingPhotos.push_back(vphotos[i]);
    testingPhotos.push_back(vphotos[i+1]);
    trainingSketches.push_back(vsketches[i]);
    testingSketches.push_back(vsketches[i+1]);
  }
  
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  LFDA lfda1(trainingPhotos, trainingSketches,16,8);
  LFDA lfda2(trainingPhotos, trainingSketches,32,16);
  
  #pragma omp parallel sections
  {
    #pragma omp section
    lfda1.compute();
    #pragma omp section
    lfda2.compute();
  }
  
  trainingPhotos.clear();
  trainingSketches.clear();
  
  vector<Mat> testingPhotosfinal(nTestingPhotos), testingSketchesfinal(nTestingSketches);
  
  cerr << "projecting testing photos" << endl;
  
  #pragma omp parallel for
  for(int i=0; i< nTestingPhotos; i++){
    testingPhotosfinal[i] = lfda1.project(testingPhotos[i]).clone();
    vconcat(testingPhotosfinal[i],lfda2.project(testingPhotos[i]).clone(),testingPhotosfinal[i]);
  }
  
  testingPhotos.clear();
  
  cerr << "projecting testing sketches" << endl;
  
  #pragma omp parallel for
  for(int i=0; i< nTestingSketches; i++){
    testingSketchesfinal[i] = lfda1.project(testingSketches[i]).clone();
    vconcat(testingSketchesfinal[i],lfda2.project(testingSketches[i]).clone(),testingSketchesfinal[i]);
  }
  
  testingSketches.clear();
  
  vector<int> rank(nTestingSketches);
  
  cerr << "calculating distances" << endl;
  
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(testingPhotosfinal[i],testingSketchesfinal[i], NORM_L2);
    //cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(testingPhotosfinal[j],testingSketchesfinal[i],NORM_L2)<= val && i!=j){
	//cerr << "small "<< j << " d1= "<< norm(testingPhotosfinal[j],testingSketchesfinal[i],NORM_L2) << endl;
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
  
  end=clock();
  tempo=(end-start)/CLOCKS_PER_SEC;
  cout << "Tempo de execução = " << tempo/60 << " min e " << tempo%60 << " s"<< endl;
  
  return 0;
  
}
