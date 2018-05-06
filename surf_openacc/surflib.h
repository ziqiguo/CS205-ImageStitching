
#ifndef SURFLIB_H
#define SURFLIB_H

#include <cv.h>
#include <iostream>
#include <highgui.h>

#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "utils.h"

#include <ctime>

using namespace std;

//! Library function builds vector of described interest points
inline void surfDetDes(IplImage *img,    /* image to find Ipoints in */
                                             std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                                             int single_mem_cpy,
                                             bool upright = false, /* run in rotation invariant mode? */
                                             int octaves = OCTAVES, /* number of octaves to calculate */
                                             int intervals = INTERVALS, /* number of intervals per octave */
                                             int init_sample = INIT_SAMPLE, /* initial sampling step */
                                             float thres = THRES /* blob response threshold */)
{
    // Create integral-image representation of the image
    IplImage *int_img = Integral(img);
    
    // Create Fast Hessian Object
    FastHessian fh(int_img, ipts, single_mem_cpy, octaves, intervals, init_sample, thres);

    // Extract interest points and store in vector ipts
    clock_t start = clock();
    fh.getIpoints();
    clock_t end = clock();
    std::cout<< "Extract Ipoint took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds" << std::endl;

    // Create Surf Descriptor Object
    Surf des(int_img, ipts);
  
    // Extract the descriptors for the ipts
    start = clock();
    des.getDescriptors(upright);
    end = clock();
    std::cout<< "Extract descriptor took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds" << std::endl;

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
