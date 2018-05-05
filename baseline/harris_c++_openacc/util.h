#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct pointData { 
    float cornerResponse;

    Point point;
};

struct by_cornerResponse { 
    bool operator()(pointData const &left, pointData const &right) { 
        return left.cornerResponse > right.cornerResponse;
    }
};

struct Derivatives {
	Mat Ix;
	Mat Iy;
	Mat Ixy;
};

class Util {
public:
	static void DisplayImage(Mat& img);
	static void DisplayMat(Mat& img);
	static void DisplayPointVector(vector<Point> vp);

	static Mat MarkInImage(Mat& img, vector<pointData> points, int radius);
};
