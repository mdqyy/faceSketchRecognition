#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "filters.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 5000;
int MAX_KERNEL_LENGTH = 31;

Mat src, dst, temp;
char window_name[] = "Filter Demo 1";

/// Function headers
int display_caption( char* caption );
int display_dst( int delay );

/**
  * function main
  */
int main( int argc, char** argv )
{
  
  vector<Mat> temp;
  Mat aux = Mat(5,5,CV_8U);
  
  for(int i=0; i<10; i++){
      randu(aux, Scalar(0), Scalar(255));
      temp.push_back(aux.clone());
  }
  
  for(auto i : temp)
    cout << i << endl;
  
  Mat vals, vecs;
  
  myPCA(temp, vecs, vals);
  
  cout << vecs << endl;
  cout << vals << endl;
  
  return 0;
  
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  
  /// Load the source image
  src = imread( "/home/marco/Imagens/0098.png", 0 );
  
  if( display_caption( "Original Image" ) != 0 ) { return 0; }
  
  dst = src.clone();
  if( display_dst( DELAY_CAPTION ) != 0 ) { return 0; }
  
  /// Applying Homogeneous blur
  //if( display_caption( "Homogeneous Blur" ) != 0 ) { return 0; }
  
  //for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
  //{ blur( src, dst, Size( i, i ), Point(-1,-1) );
  //if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }
  
  /// Applying Gaussian blur
  //if( display_caption( "Gaussian Blur" ) != 0 ) { return 0; }
  
  //for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
  //{ GaussianBlur( src, dst, Size( i, i ), 0, 0 );
  //if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }
  
  /// Applying Gaussian
  if( display_caption( "Gaussian" ) != 0 ) { return 0; }
  
  dst=GaussianFilter(src);
  if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
  
  /// Applying Difference of Gaussian
  if( display_caption( "DoG" ) != 0 ) { return 0; }
  
  dst=DoGFilter(src);
  if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
  
  /// Applying CSDN
  if( display_caption( "CSDN" ) != 0 ) { return 0; }
  
  dst=CSDNFilter(src);
  if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
  
  /// Applying Median blur
  //if( display_caption( "Median Blur" ) != 0 ) { return 0; }
  
  //for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
  //{ medianBlur ( src, dst, i );
  //if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }
  
  /// Applying Bilateral Filter
  //if( display_caption( "Bilateral Blur" ) != 0 ) { return 0; }
  
  //for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
  //{ bilateralFilter ( src, dst, i, i*2, i/2 );
  //if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }
  
  /// Wait until user press a key
  display_caption( "End: Press a key!" );
  
  waitKey(0);
  return 0;
}

int display_caption( char* caption )
{
  dst = Mat::zeros( src.size(), src.type() );
  putText( dst, caption,
	    Point( src.cols/4, src.rows/2),
	    CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255) );
  
  imshow( window_name, dst );
  int c = waitKey( DELAY_CAPTION );
  if( c >= 0 ) { return -1; }
  return 0;
}

int display_dst( int delay )
{
  //cout << "size = "<< dst.size() << ", type = " << dst.type() << endl;
  imshow( window_name, dst );
  int c = waitKey ( delay );
  if( c >= 0 ) { return -1; }
  return 0;
}