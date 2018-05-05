#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "harris.h"

using namespace cv;
using namespace std;

//Harris algorithm parameters
// Specifies the sensitivity factor of the Harris algorithm (0 < k < 0.25)
float k = 0.25;
// Size of the box filter that is applied to the integral images
int boxFilterSize = 3;
// dimension of the maxima suppression box around a maxima
int maximaSuppressionDimension = 10;

//UI parameters
// dimension of the objects showing a maxima in the image
int markDimension = 5;
// constant for the slider-value division
float divisionConstant = 1000000;

//Global variables
bool gauss = true;
Mat m_img;

void doHarris() {
    // compute harris
    clock_t start;

    start = clock();
    Harris harris(m_img, k, boxFilterSize, gauss);
    cout << "Harris total: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    // get vector of points wanted
    start = clock();
    vector<pointData> resPts = harris.getMaximaPoints(0.000534, boxFilterSize, maximaSuppressionDimension);
    cout << "Get maxima Points: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;    
    // cout << resPts.size() << " Points" << endl;

    Mat _img = Util::MarkInImage(m_img, resPts, markDimension);
    imshow("HarrisCornerDetector", _img);
}


//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // read image from file + error handling
    Mat img;

    if (argc == 1) {
        cout << "No image provided! Usage: ./Ex1 [path to image]" << endl << "Using default image: haus.jpg" << endl;

        img = imread("test_data/haus.jpg");
    } else {
        img = imread(argv[1]);
    }

    // if(img.rows > 100 || img.cols > 100) {
    //     int newrows = 600;
    //     int newcols = img.cols * newrows / img.rows;

    //     resize(img, img, Size(newcols, newrows), 0, 0, INTER_CUBIC);
    // }
    img.copyTo(m_img);

    // create UI and show the image
    namedWindow("HarrisCornerDetector", 1);


    clock_t start = clock();
    doHarris();
    cout << "Overall: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;


    // imshow("HarrisCornerDetector", img);
    waitKey(0);

    return 0;

}
