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
  loadImages(argv[3],photos,1);
  loadImages(argv[4],sketches,1);
  loadImages(argv[5],extra,1);
  
  
  testingPhotos.insert(testingPhotos.end(),vphotos.begin()+471,vphotos.begin()+571);
  testingSketches.insert(testingSketches.end(),vsketches.begin()+471,vsketches.begin()+571);
  //testingPhotos.insert(testingPhotos.end(), extra.begin(), extra.end());
  
  //for(int i=0; i<10; i++)
    //testingPhotos.insert(testingPhotos.end(), extra.begin(), extra.end());
  
  trainingPhotos.insert(trainingPhotos.end(),vphotos.begin()+383,vphotos.begin()+471);
  trainingSketches.insert(trainingSketches.end(),vsketches.begin()+383,vsketches.begin()+471);
  
  int nTestingSketches = testingSketches.size(),
  nTestingPhotos = testingPhotos.size(),
  nTraining = trainingPhotos.size();
  
  cerr << nTraining << " pares para treinamento." << endl;
  cerr << nTestingSketches << " sketches para reconhecimento." << endl;
  cerr << nTestingPhotos << " fotos na galeria." << endl;
  
  LFDA lfda(trainingPhotos, trainingSketches);
  lfda.compute();
     
  vector<Mat> testingPhotosfinal2, testingSketchesfinal2;
  
  cerr << "projecting testing photos" << endl;
  
  for(int i=0; i< nTestingPhotos; i++)
    testingPhotosfinal2.push_back(lfda.project(testingPhotos[i]));
  
  cerr << "projecting testing sketches" << endl;
  
  for(int i=0; i< nTestingSketches; i++)
    testingSketchesfinal2.push_back(lfda.project(testingSketches[i]));
  
  vector<int> rank2(nTestingSketches);
  
  cerr << "calculating distances" << endl;
  
  for(int i=0; i<nTestingSketches; i++){
    double val = norm(testingPhotosfinal2[i],testingSketchesfinal2[i], NORM_L2);//euclideanDistance(testingPhotosfinal2[i],testingSketchesfinal2[i]);
    if(val!=val)
      cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nTestingPhotos; j++){
      if(norm(testingPhotosfinal2[j],testingSketchesfinal2[i],NORM_L2)<= val && i!=j){//euclideanDistance(testingPhotosfinal2[j],testingSketchesfinal2[i])<= val && i!=j){
	if(val!=val)
	  cerr << "small "<< i << " d1= "<< val << endl;
	temp++;
      }
    }
    rank2[i] = temp;
    cerr << i << " rank= " << temp << endl;
  }
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
  cerr << "Rank "<< i << ": ";
  cerr << "d1= " << (float)count_if(rank2.begin(), rank2.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }

  end=clock();
  tempo=(end-start)/CLOCKS_PER_SEC;
  cout << "Tempo de execução = " << tempo/60 << " min e " << tempo%60 << " s"<< endl;
  return 0;
  
  
  //LFDA
  
  //Patches
  vector<vector<vector<Mat> > > trainingPhotosDesc, trainingSketchesDesc,
  testingPhotosDesc, testingSketchesDesc;
  
  //Extraindo os patches das imagens de treinamento
  for(int i=0; i < nTraining; i++){
    vector<vector<Mat> > photoV, sketchV;
    vector<vector<Mat> > photoPatches, sketchPatches;
    patcher(trainingPhotos[i],16,8,photoPatches);
    patcher(trainingSketches[i],16,8,sketchPatches);
    trainingPhotosDesc.push_back(photoV);
    trainingSketchesDesc.push_back(sketchV);
    for(uint j=0; j<photoPatches.size(); j++){
      vector<Mat> photoCol, sketchCol;
      trainingPhotosDesc[i].push_back(photoCol);
      trainingSketchesDesc[i].push_back(sketchCol);
      for(uint k=0; k<photoPatches[0].size(); k++){
	//cerr << i << " " << j << " " << k << endl;
	Mat a, b, desc1, desc2;
	calcSIFTDescriptors(photoPatches[j][k],a);
	calcLBPHistogram(photoPatches[j][k],b);
	hconcat(a,b,desc1);
	trainingPhotosDesc[i][j].push_back(desc1);
	calcSIFTDescriptors(sketchPatches[j][k],a);
	calcLBPHistogram(sketchPatches[j][k],b);
	hconcat(a,b,desc2);
	trainingSketchesDesc[i][j].push_back(desc2);
      }
    }
    if(i==0)
      cerr << "Quantidade de patches= " << photoPatches.size() <<
      "x" << photoPatches[0].size() << " de " << photoPatches[0][0].size() << endl;
  }
  
  cerr << "Descritores dos patches de treinamento= " << trainingPhotosDesc.size() <<
  " imgs, divididas em " << trainingPhotosDesc[0].size() <<
  "x" << trainingPhotosDesc[0][0].size() <<
  " de " << trainingPhotosDesc[0][0][0].size() << endl;
    
  cerr << "Extraindo descritores dos patches de teste= ";
  
  //Extraindo os patches das fotos de teste
  for(int i=0; i < nTestingPhotos; i++){
    vector<vector<Mat> > photoV;
    vector<vector<Mat> > photoPatches;
    patcher(testingPhotos[i],16,8,photoPatches);
    testingPhotosDesc.push_back(photoV);
    for(uint j=0; j<photoPatches.size(); j++){
      vector<Mat> photoCol;
      testingPhotosDesc[i].push_back(photoCol);
      for(uint k=0; k<photoPatches[0].size(); k++){
	//cerr << i << " " << j << " " << k << endl;
	Mat a, b, desc1;
	calcSIFTDescriptors(photoPatches[j][k],a);
	calcLBPHistogram(photoPatches[j][k],b);
	hconcat(a,b,desc1);
	testingPhotosDesc[i][j].push_back(desc1);
      }
    }
    if(!(i%50))
      cerr << i*100.0/nTestingPhotos << "%" << endl;
  }
  
  //Extraindo os patches dos sketches de teste
  for(int i=0; i < nTestingSketches; i++){
    vector<vector<Mat> > sketchV;
    vector<vector<Mat> > sketchPatches;
    patcher(testingSketches[i],16,8,sketchPatches);
    testingSketchesDesc.push_back(sketchV);
    for(uint j=0; j<sketchPatches.size(); j++){
      vector<Mat> sketchCol;
      testingSketchesDesc[i].push_back(sketchCol);
      for(uint k=0; k<sketchPatches[0].size(); k++){
	//cerr << i << " " << j << " " << k << endl;
	Mat a, b, desc1;
	calcSIFTDescriptors(sketchPatches[j][k],a);
	calcLBPHistogram(sketchPatches[j][k],b);
	hconcat(a,b,desc1);
	testingSketchesDesc[i][j].push_back(desc1);
      }
    }
  }
  
  cerr << "Descritores dos patches de teste= " << testingPhotosDesc.size() <<
  " imgs, divididas em " << testingPhotosDesc[0].size() <<
  "x" << testingPhotosDesc[0][0].size() <<
  " de " << testingPhotosDesc[0][0][0].size() << endl;
  
  // Criando os rótulos da LDA
  vector<int> _classes;
  for(int i=0; i < nTraining; i++){
    _classes.push_back(i);
  }
  for(int i=0; i < nTraining; i++){
    _classes.push_back(i);
  }
  
  int n = trainingSketchesDesc[0].size()*trainingSketchesDesc[0][0][0].cols;
  cerr << "Tamanho do slice= " << trainingSketchesDesc[0].size() << "x"
  << trainingSketchesDesc[0][0][0].cols << "= " << n << endl;
  
  //*Desc[imagem][linha][coluna]
  //       0-87    0-29   0-23
  
  vector<Mat> _dataTrainingCols, _dataTestingPhotosCols, _dataTestingSketchesCols;
  
  //Alocando os slices do conjunto de treinamento
  for(uint k=0; k<trainingPhotosDesc[0][0].size();k++){
    Mat _data;
    for(uint i=0; i< trainingPhotosDesc.size(); i++){
      Mat temp = trainingPhotosDesc[i][0][k].clone();
      for(uint j=1; j<trainingPhotosDesc[0].size();j++){
	hconcat(temp,trainingPhotosDesc[i][j][k],temp);
      }
      normalize(temp,temp,1);
      _data.push_back(temp);
    }
    for(uint i=0; i< trainingSketchesDesc.size(); i++){
      Mat temp = trainingSketchesDesc[i][0][k].clone();
      for(uint j=1; j<trainingSketchesDesc[0].size();j++){
	hconcat(temp,trainingSketchesDesc[i][j][k],temp);
      }
      normalize(temp,temp,1);
      _data.push_back(temp);
    }
    _dataTrainingCols.push_back(_data);
  }
  
  cerr << "Tamanho do vetor de slices de treinamento= " << _dataTrainingCols.size()
  << " de " << _dataTrainingCols[0].size() << endl;
  
  //Alocando os slices das fotos de teste
  for(uint k=0; k<testingPhotosDesc[0][0].size();k++){
    Mat _data;
    for(uint i=0; i< testingPhotosDesc.size(); i++){
      Mat temp = testingPhotosDesc[i][0][k].clone();
      for(uint j=1; j<testingPhotosDesc[0].size();j++){
	hconcat(temp,testingPhotosDesc[i][j][k],temp);
      }
      normalize(temp,temp,1);
      _data.push_back(temp);
    }
    _dataTestingPhotosCols.push_back(_data);
  }
  
  cerr << "Tamanho do vetor de slices de fotos de teste= "
  << _dataTestingPhotosCols.size()
  << " de " << _dataTestingPhotosCols[0].size() << endl;
  
  
  //Alocando os slices dos sketches de teste
  for(uint k=0; k<testingSketchesDesc[0][0].size();k++){
    Mat _data;
    for(uint i=0; i< testingSketchesDesc.size(); i++){
      Mat temp = testingSketchesDesc[i][0][k].clone();
      for(uint j=1; j<testingSketchesDesc[0].size();j++){
	hconcat(temp,testingSketchesDesc[i][j][k],temp);
      }
      normalize(temp,temp,1);
      _data.push_back(temp);
    }
    _dataTestingSketchesCols.push_back(_data);
  }
  
  cerr << "Tamanho do vetor de slices de sketches de teste= "
  << _dataTestingSketchesCols.size()
  << " de " << _dataTestingSketchesCols[0].size() << endl;
  
  vector<vector<Mat> > photosFinalDescriptors(nTestingPhotos),
  sketchesFinalDescriptors(nTestingSketches);
  //FinalDescriptor[imagem][coluna]
  
  int num_pca = 100<=nTraining-1 ? 100 : nTraining-1;
  
  for(uint k=0; k<_dataTrainingCols.size(); k++){//cada coluna
    
    PCA pca(_dataTrainingCols[k],Mat(),CV_PCA_DATA_AS_ROW,num_pca); //<< olhar isso;
    Mat _dataPCA;
    pca.project(_dataTrainingCols[k], _dataPCA);
    
    cerr << _dataPCA.size() << endl;
    cerr << _classes.size() << endl;
    cerr << "Processando LDA " << k << endl;
    
    LDA lda;
    lda.compute(_dataPCA,_classes);
    
    Mat temp1, aux1;
    pca.project(_dataTestingPhotosCols[k], temp1);
    for(int i=0; i<temp1.rows; i++){
      aux1= lda.project(temp1.row(i));
      //aux1*=1.0/(abs(12-k)+1);
      photosFinalDescriptors[i].push_back(aux1);
    }
    
    Mat temp2, aux2;
    pca.project(_dataTestingSketchesCols[k], temp2);
    for(int i=0; i<temp2.rows; i++){
      aux2 = lda.project(temp2.row(i));
      //aux2*=1.0/(abs(12-k)+1);
      sketchesFinalDescriptors[i].push_back(aux2);
    }
  }
  
  vector<int> rank(nTestingSketches);
  for(int i=0; i< nTestingSketches; i++){
    float val=0;
    for(uint k=0; k<photosFinalDescriptors[i].size(); k++){
      val+= euclideanDistance(photosFinalDescriptors[i][k],
			      sketchesFinalDescriptors[i][k]);
      //cerr << photosFinalDescriptors[i][k] << endl;
      //cerr << sketchesFinalDescriptors[i][k] << endl;
    }
    
    cerr << "sketch " << i << " = " << val << endl;
    
    int count=0;
    for(int j=0; j<nTestingPhotos; j++){
      float temp=0;
      for(uint k=0; k<photosFinalDescriptors[i].size(); k++){
	temp+= euclideanDistance(photosFinalDescriptors[j][k],
				 sketchesFinalDescriptors[i][k]);
      }
      if(temp<=val && j!=i){
	count++;
	cerr << " ... sketch " << i << " = " << val << " with " << j << endl;
      }
      if(temp!=temp || val!=val){
	cerr << "NaN" << endl;
	return -1;
      }
    }
    rank[i]=count;
  }
  
  for(int i=0; i<nTestingSketches; i++)
    cerr << i << " rank= " << rank[i] << endl;
  
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50})
  {
  cerr << "Rank "<< i << ": ";
  cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
  }

  
  return 0;
}

