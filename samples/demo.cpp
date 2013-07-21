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
#include "eigenTransformation.hpp"
#include "utils.hpp"
#include "lfda.hpp"

using namespace cv;
using namespace std;

////Global variables
bool Rank1(float i) {return i==0;}
bool Rank2(float i) {return i<2;}
bool Rank3(float i) {return i<3;}
bool Rank4(float i) {return i<4;}
bool Rank5(float i) {return i<5;}
bool Rank6(float i) {return i<6;}
bool Rank7(float i) {return i<7;}
bool Rank8(float i) {return i<8;}
bool Rank9(float i) {return i<9;}
bool Rank10(float i) {return i<10;}
bool Rank20(float i) {return i<20;}
bool Rank50(float i) {return i<50;}
bool Rank100(float i) {return i<100;}

int main(int argc, char** argv){

	/*	if(argc != 5){
		cout << "Usage: "<< argv[0] << "<photos> <sketches>"
				<< endl;
		return -1;
	}
	 */


	Mat avgPhoto, avgSketch, eigenPhotos, eigenSketches,
	pPhotos, pSketches, vecsPhotos, vecsSketches, valsDiagPhotos, valsDiagSketches;

	vector<Mat> trainingPhotos,	trainingSketches,
	testingPhotos,testingSketches;

	vector<Mat> photos, sketches, extra, vphotos, vsketches;

	loadImages(argv[1],photos,1);
	loadImages(argv[2],sketches,1);
	//loadImages(argv[3],extra,1);
	loadImages(argv[3],vphotos,1);
	loadImages(argv[4],vsketches,1);

	testingPhotos.insert(testingPhotos.end(),photos.begin(),photos.end());
	testingSketches.insert(testingSketches.end(),sketches.begin(),sketches.end());

	trainingPhotos.insert(trainingPhotos.end(),vphotos.begin(),vphotos.begin()+100);
	trainingSketches.insert(trainingSketches.end(),vsketches.begin(),vsketches.begin()+100);

	//testingPhotos.insert(testingPhotos.end(), extra.begin(), extra.begin()+999);

	int nTestingSketches = testingSketches.size(),
			nTestingPhotos = testingPhotos.size(),
			nTraining = trainingPhotos.size();

	cerr << nTraining << " pares para treinamento." << endl;
	cerr << nTestingSketches << " sketches para reconhecimento." << endl;
	cerr << nTestingPhotos << " fotos na galeria." << endl;


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
		for(int j=0; j<photoPatches.size(); j++){
			vector<Mat> photoCol, sketchCol;
			trainingPhotosDesc[i].push_back(photoCol);
			trainingSketchesDesc[i].push_back(sketchCol);
			for(int k=0; k<photoPatches[0].size(); k++){
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
		for(int j=0; j<photoPatches.size(); j++){
			vector<Mat> photoCol;
			testingPhotosDesc[i].push_back(photoCol);
			for(int k=0; k<photoPatches[0].size(); k++){
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
		for(int j=0; j<sketchPatches.size(); j++){
			vector<Mat> sketchCol;
			testingSketchesDesc[i].push_back(sketchCol);
			for(int k=0; k<sketchPatches[0].size(); k++){
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
	for(int k=0; k<trainingPhotosDesc[0][0].size();k++){
		Mat _data;
		for(int i=0; i< trainingPhotosDesc.size(); i++){
			Mat temp = trainingPhotosDesc[i][0][k].clone();
			for(int j=1; j<trainingPhotosDesc[0].size();j++){
				hconcat(temp,trainingPhotosDesc[i][j][k],temp);
			}
			//temp = temp*(30.0/sum(temp).val[0]);
			_data.push_back(temp);
		}
		for(int i=0; i< trainingSketchesDesc.size(); i++){
			Mat temp = trainingSketchesDesc[i][0][k].clone();
			for(int j=1; j<trainingSketchesDesc[0].size();j++){
				hconcat(temp,trainingSketchesDesc[i][j][k],temp);
			}
			//temp = temp*(30.0/sum(temp).val[0]);
			_data.push_back(temp);
		}
		_dataTrainingCols.push_back(_data);
	}

	cerr << "Tamanho do vetor de slices de treinamento= " << _dataTrainingCols.size()
																	<< " de " << _dataTrainingCols[0].size() << endl;

	//Alocando os slices das fotos de teste
	for(int k=0; k<testingPhotosDesc[0][0].size();k++){
		Mat _data;
		for(int i=0; i< testingPhotosDesc.size(); i++){
			Mat temp = testingPhotosDesc[i][0][k].clone();
			for(int j=1; j<testingPhotosDesc[0].size();j++){
				hconcat(temp,testingPhotosDesc[i][j][k],temp);
			}
			//temp = temp*(30.0/sum(temp).val[0]);
			_data.push_back(temp);
		}
		_dataTestingPhotosCols.push_back(_data);
	}

	cerr << "Tamanho do vetor de slices de fotos de teste= "
			<< _dataTestingPhotosCols.size()
			<< " de " << _dataTestingPhotosCols[0].size() << endl;


	//Alocando os slices dos sketches de teste
	for(int k=0; k<testingSketchesDesc[0][0].size();k++){
		Mat _data;
		for(int i=0; i< testingSketchesDesc.size(); i++){
			Mat temp = testingSketchesDesc[i][0][k].clone();
			for(int j=1; j<testingSketchesDesc[0].size();j++){
				hconcat(temp,testingSketchesDesc[i][j][k],temp);
			}
			//temp = temp*(30.0/sum(temp).val[0]);
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

	for(int k=0; k<_dataTrainingCols.size(); k++){//cada coluna

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
		for(int k=0; k<photosFinalDescriptors[i].size(); k++){
			val+= euclideanDistance(photosFinalDescriptors[i][k],
					sketchesFinalDescriptors[i][k]);
			//cerr << photosFinalDescriptors[i][k] << endl;
			//cerr << sketchesFinalDescriptors[i][k] << endl;
		}

		cerr << "sketch " << i << " = " << val << endl;

		int count=0;
		for(int j=0; j<nTestingPhotos; j++){
			float temp=0;
			for(int k=0; k<photosFinalDescriptors[i].size(); k++){
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

	cerr << "LFDA" << endl;
	cerr << "Rank 1: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank1)/testingSketches.size()<< endl;
	cerr << "Rank 2: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank2)/testingSketches.size()<< endl;
	cerr << "Rank 3: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank3)/testingSketches.size()<< endl;
	cerr << "Rank 4: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank4)/testingSketches.size()<< endl;
	cerr << "Rank 5: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank5)/testingSketches.size()<< endl;
	cerr << "Rank 6: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank6)/testingSketches.size()<< endl;
	cerr << "Rank 7: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank7)/testingSketches.size()<< endl;
	cerr << "Rank 8: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank8)/testingSketches.size()<< endl;
	cerr << "Rank 9: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank9)/testingSketches.size()<< endl;
	cerr << "Rank 10: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank10)/testingSketches.size()<< endl;
	cerr << "Rank 50: ";
	cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), Rank50)/testingSketches.size()<< endl;


	return 0;

	//EigenTransformation

	cerr << "creating eigenphotos" << endl;
	createEigenSpace(trainingPhotos,pPhotos,avgPhoto,vecsPhotos,valsDiagPhotos,eigenPhotos);
	cerr << "creating eigensketches" << endl;
	createEigenSpace(trainingSketches,pSketches,avgSketch,vecsSketches,valsDiagSketches,eigenSketches);

	vector<Mat> photosB(nTestingPhotos), sketchesB(nTestingSketches),
			photosContr(nTestingPhotos), sketchesContr(nTestingSketches),
			recPhotosB(nTestingSketches), recSketchesB(nTestingPhotos);

	Mat image;

	cerr << "projecting testing sketches" << endl;
	for(int i=0; i<nTestingSketches; i++){
		image = testingSketches[i].clone();
		image.convertTo(image, CV_32F);
		image -= avgSketch;
		image = image.reshape(1, eigenSketches.rows);
		sketchesB[i] = eigenSketches.t()*image;
		sketchesContr[i] = vecsSketches*valsDiagSketches*sketchesB[i];
		sketchesContr[i] -= ((float)sum(sketchesContr[i])[0] - 1)/nTraining;
		recPhotosB[i] = eigenPhotos.t()*pPhotos*sketchesContr[i];
	}

	cerr << "projecting testing photos" << endl;
	for(int i=0; i<nTestingPhotos; i++){
		image = testingPhotos[i].clone();
		image.convertTo(image, CV_32F);
		image -= avgPhoto;
		image = image.reshape(1, eigenPhotos.rows);
		photosB[i] = eigenPhotos.t()*image;
		photosContr[i] = vecsPhotos*valsDiagPhotos*photosB[i];
		photosContr[i] -= ((float)sum(photosContr[i])[0] - 1)/nTraining;
		recSketchesB[i] = eigenSketches.t()*pSketches*photosContr[i];
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


	cerr << "Rank 1: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank1)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank1)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank1)/nTestingSketches << endl;
	cerr << "Rank 2: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank2)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank2)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank2)/nTestingSketches << endl;
	cerr << "Rank 3: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank3)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank3)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank3)/nTestingSketches << endl;
	cerr << "Rank 4: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank4)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank4)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank4)/nTestingSketches << endl;
	cerr << "Rank 5: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank5)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank5)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank5)/nTestingSketches << endl;
	cerr << "Rank 6: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank6)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank6)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank6)/nTestingSketches << endl;
	cerr << "Rank 7: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank7)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank7)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank7)/nTestingSketches << endl;
	cerr << "Rank 8: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank8)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank8)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank8)/nTestingSketches << endl;
	cerr << "Rank 9: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank9)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank9)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank9)/nTestingSketches << endl;
	cerr << "Rank 10: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank10)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank10)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank10)/nTestingSketches << endl;
	cerr << "Rank 50: ";
	cerr << "d1= " << (float)count_if(d1.begin(), d1.end(), Rank50)/nTestingSketches;
	cerr << " ,d2= " << (float)count_if(d2.begin(), d2.end(), Rank50)/nTestingSketches;
	cerr << " ,d3= " << (float)count_if(d3.begin(), d3.end(), Rank50)/nTestingSketches << endl;



	return 0;
}

