#include "util.h"

void Util::DisplayImage(Mat& img) {
	// Create UI
    namedWindow("ImgViewer", 1);

    imshow("ImgViewer", img);

    // Prevents program from exiting
    waitKey(0);
}

//-----------------------------------------------------------------------------------------------
void Util::DisplayMat(Mat& img) {
	cout << "-= Display Matrix =-";

	int rowCount = 0;

    for (int r = 0; r < img.rows; r++) {
    	cout << endl;

    	int colCount = 0;

        for (int c = 0; c < img.cols; c++) {
            cout << img.at<float>(r,c) << ", ";
            colCount++;
        }

        rowCount++;

        cout << " -> " << colCount << "cols";
    }

    cout << "-> " << rowCount << "rows" << endl;
}

//-----------------------------------------------------------------------------------------------
void Util::DisplayPointVector(vector<Point> vp) {
    vector<Point>::iterator pIterator;

    for(int i=0; i<vp.size(); i++) {
        Point p = vp[i];

        cout << p.x << "," << p.y << "; ";
    }
}

//-----------------------------------------------------------------------------------------------
Mat Util::MarkInImage(Mat& img, vector<pointData> points, int radius) {
	Mat retImg;
	img.copyTo(retImg);

	for(vector<pointData>::iterator it = points.begin(); it != points.end(); ++it) {
		Point center = (*it).point;

		// down
		for(int r=-radius; r<radius; r++) {
			retImg.at<Vec3b>(Point(center.y+r,center.x+radius)) = Vec3b(0, 0, 255);
		}

		// up
		for(int r=-radius; r<radius; r++) {
			retImg.at<Vec3b>(Point(center.y+r,center.x-radius)) = Vec3b(0, 0, 255);
		}

		// left
		for(int c=-radius; c<radius; c++) {
			retImg.at<Vec3b>(Point(center.y-radius,center.x+c)) = Vec3b(0, 0, 255);
		}

		// right
		for(int c=-radius; c<radius; c++) {
			retImg.at<Vec3b>(Point(center.y+radius,center.x+c)) = Vec3b(0, 0, 255);
		}

		retImg.at<Vec3b>(Point(center.y,center.x)) = Vec3b(0, 255, 0);
	}

	return retImg;
}
