
#ifndef SURFLIB_H
#define SURFLIB_H

#include <cv.h>
#include <highgui.h>
#include <ctime>
#include <iostream>

#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "utils.h"


//! Library function builds vector of described interest points
inline void surfDetDes(IplImage *img,    /* image to find Ipoints in */
                                             std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                                             bool upright = false, /* run in rotation invariant mode? */
                                             int octaves = OCTAVES, /* number of octaves to calculate */
                                             int intervals = INTERVALS, /* number of intervals per octave */
                                             int init_sample = INIT_SAMPLE, /* initial sampling step */
                                             float thres = THRES /* blob response threshold */)
{
    // Create integral-image representation of the image
    IplImage *int_img = Integral(img);
    
    // Create Fast Hessian Object
    FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);
    
    // Extract interest points and store in vector ipts
    clock_t t0 = clock();
    fh.getIpoints();
    clock_t t1 = clock();
    std::cout<< "Keypoint detection took: " << float(t1 - t0) / CLOCKS_PER_SEC    << " seconds" << std::endl;
    
    // Create Surf Descriptor Object
    Surf des(int_img, ipts);

    // Extract the descriptors for the ipts
    clock_t t2 = clock();
    des.getDescriptors(upright);
    clock_t t3 = clock();
    std::cout<< "Keypoint description took: " << float(t3 - t2) / CLOCKS_PER_SEC    << " seconds" << std::endl;

    // Deallocate the integral image
    cvReleaseImage(&int_img);
}


//! Library function builds vector of interest points
inline void surfDet(IplImage *img,    /* image to find Ipoints in */
                                        std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                                        int octaves = OCTAVES, /* number of octaves to calculate */
                                        int intervals = INTERVALS, /* number of intervals per octave */
                                        int init_sample = INIT_SAMPLE, /* initial sampling step */
                                        float thres = THRES /* blob response threshold */)
{
    // Create integral image representation of the image
    IplImage *int_img = Integral(img);

    // Create Fast Hessian Object
    FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

    // Extract interest points and store in vector ipts
    fh.getIpoints();

    // Deallocate the integral image
    cvReleaseImage(&int_img);
}




//! Library function describes interest points in vector
inline void surfDes(IplImage *img,    /* image to find Ipoints in */
                                        std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                                        bool upright = false) /* run in rotation invariant mode? */
{ 
    // Create integral image representation of the image
    IplImage *int_img = Integral(img);

    // Create Surf Descriptor Object
    Surf des(int_img, ipts);

    // Extract the descriptors for the ipts
    des.getDescriptors(upright);
    
    // Deallocate the integral image
    cvReleaseImage(&int_img);
}


#endif
