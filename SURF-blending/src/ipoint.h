#ifndef IPOINT_H
#define IPOINT_H

#include <vector>
#include <math.h>
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


//-------------------------------------------------------

class Ipoint; // Pre-declaration
typedef std::vector<Ipoint> IpVec;
typedef std::vector<std::pair<Ipoint, Ipoint> > IpPairVec;

//-------------------------------------------------------

//! Ipoint operations
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches);
// void getMatchesKDTree(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches);
cv::Mat getCvWarpped(IpPairVec &matches, IplImage *original);
cv::Mat getCvStitch(IplImage *src, cv::Mat warpped);
cv::Mat getWarpped(IpPairVec &matches, IplImage *original);//, cv::detail::MultiBandBlender &blender);
//cv::Mat getWarppedReMap(IpPairVec &matches, IplImage *original);
cv::Mat getHomography(IpPairVec &matches, IplImage *original);
cv::Mat getBlended(IplImage *img1, IplImage *img2, IpPairVec &matches);
// cv::Point2f translateCorner(cv::Mat H, const cv::Point src_corners);
//-------------------------------------------------------

class Ipoint {

public:

    //! Destructor
    ~Ipoint() {};

    //! Constructor
    Ipoint() : orientation(0) {};

    //! Gets the distance in descriptor space between Ipoints
    float operator-(const Ipoint &rhs)
    {
        float sum=0.f;
        for(int i=0; i < 64; ++i)
            sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
        return sqrt(sum);
    };

    //! Coordinates of the detected interest point
    float x, y;

    //! Detected scale
    float scale;

    //! Orientation measured anti-clockwise from +ve x-axis
    float orientation;

    //! Sign of laplacian for fast matching purposes
    int laplacian;

    //! Vector of descriptor components
    float descriptor[64];

    //! Placeholds for point motion (can be used for frame to frame motion analysis)
    float dx, dy;

    //! Used to store cluster index
    int clusterIndex;
};

//-------------------------------------------------------


#endif
