#include "surflib.h"
// #include "kmeans.h"
#include <ctime>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
//https://gist.github.com/yoggy/3246274

/*
    Usage: 1 -> regular stitching 2 images
           3 -> use blender to stitch

    get Ipts took: 3.54182 seconds
    extract features took: 1.30219 seconds
    get Ipts took: 3.55647 seconds
    extract features took: 1.46639 seconds
    OpenSURF took: 3.03159 seconds to get matches.
    

    (warpping took: 1.28497
    stitching took: 0.057989)

    find homography took: 0.003098 seconds.
    warpping took: 0.389348
    creating mask took: 0.262575
    blending took: 0.66731
*/


int mainImage(void)
{
    // Declare Ipoints and other stuff
    IpVec ipts;
    IplImage *img=cvLoadImage("../imgs/sf.jpg");

    // Detect and describe interest points in the image
    clock_t start = clock();
    surfDetDes(img, ipts, false, 5, 4, 2, 0.0004f); 
    clock_t end = clock();

    std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
    std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds" << std::endl;

    // Draw the detected points
    drawIpoints(img, ipts);
    
    // Display the result
    showImage(img);

    return 0;
}

//-------------------------------------------------------

int mainVideo(void)
{
    // Initialise capture device
    CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
    if(!capture) error("No Capture");

    // Initialise video writer
    //cv::VideoWriter vw("c:\\out.avi", CV_FOURCC('D','I','V','X'),10,cvSize(320,240),1);
    //vw << img;

    // Create a window 
    cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

    // Declare Ipoints and other stuff
    IpVec ipts;
    IplImage *img=NULL;

    // Main capture loop
    while( 1 ) 
    {
        // Grab frame from the capture source
        img = cvQueryFrame(capture);

        // Extract surf points
        surfDetDes(img, ipts, false, 4, 4, 2, 0.004f);        

        // Draw the detected points
        drawIpoints(img, ipts);

        // Draw the FPS figure
        drawFPS(img);

        // Display the result
        cvShowImage("OpenSURF", img);

        // If ESC key pressed exit loop
        if( (cvWaitKey(10) & 255) == 27 ) break;
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow( "OpenSURF" );
    return 0;
}


int mainStaticMatch()
{
    IplImage *img1 = cvLoadImage("../../sequential/img1.jpg");
    IplImage *img2 = cvLoadImage("../../sequential/img2.jpg");

    IpVec ipts1, ipts2;

    //clock_t start = clock();
    surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
    surfDetDes(img2,ipts2,false,4,4,2,0.0001f);
    //clock_t end = clock();
    //std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds to find Ipts and their FDes." << std::endl;
    IpPairVec matches;
    clock_t start = clock();
    //getMatchesKDTree(ipts1, ipts2, matches);
    getMatches(ipts1, ipts2, matches);
    clock_t end = clock();
    std::cout << "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC << " seconds to get matches." << std::endl;

    // start = clock();
    // cv::Mat warpped = getCvWarpped(matches, img2);
    // end = clock();
    // std::cout<< "compute H, warpping took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
    // start = clock();
    // cv::Mat stitched = getCvStitch(img1, warpped);
    // end = clock();
    // std::cout<< "stitching took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
    // cv::Mat warpped = getWarppedReMap(matches, img2);

    // cv::Mat H = getHomography(matches, img2);
    // makePanorama(img1, img2, H, std::vector<std::vector<Point>> corners);



    start = clock();
    cv::Mat warpped = getWarpped(matches, img2);
    end = clock();
    std::cout<< "warpping took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    start = clock();
    cv::Mat stitched = getCvStitch(img1, warpped);
    end = clock();
    std::cout<< "stitching took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
 
    
    //cv::namedWindow("stitched", CV_WINDOW_AUTOSIZE );
    //cv::namedWindow("1", CV_WINDOW_AUTOSIZE );
    //cv::namedWindow("2", CV_WINDOW_AUTOSIZE );
    cvShowImage("1", img1);
    cvShowImage("2", img2);
    //cv::imshow("warp", warpped);  
    //cv::imshow("mask", mask2);
    cv::imshow("stitched", stitched);
    cvWaitKey(0);

    return 0;
}

int mainBlending()
{
    IplImage *img1 = cvLoadImage("../../sequential/img1.jpg");
    IplImage *img2 = cvLoadImage("../../sequential/img2.jpg");

    IpVec ipts1, ipts2;
    surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
    surfDetDes(img2,ipts2,false,4,4,2,0.0001f);
    IpPairVec matches;
    clock_t start = clock();
    getMatches(ipts1, ipts2, matches);
    clock_t end = clock();
    std::cout << "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC << " seconds to get matches." << std::endl;
    cv::Mat result_s = getBlended(img1, img2, matches);
    cv::imshow("blended", result_s);
    cvWaitKey(0);

    return 0;
}

//-------------------------------------------------------

// int mainKmeans(void)
// {
//     IplImage *img = cvLoadImage("imgs/img1.jpg");
//     IpVec ipts;
//     Kmeans km;
    
//     // Get Ipoints
//     surfDetDes(img,ipts,true,3,4,2,0.0006f);

//     for (int repeat = 0; repeat < 10; ++repeat)
//     {

//         IplImage *img = cvLoadImage("imgs/img1.jpg");
//         km.Run(&ipts, 5, true);
//         drawPoints(img, km.clusters);

//         for (unsigned int i = 0; i < ipts.size(); ++i)
//         {
//             cvLine(img, cvPoint(ipts[i].x,ipts[i].y), cvPoint(km.clusters[ipts[i].clusterIndex].x ,km.clusters[ipts[i].clusterIndex].y),cvScalar(255,255,255));
//         }

//         showImage(img);
//     }

//     return 0;
// }

//-------------------------------------------------------

int main(int argc, char* argv[]) 
{
    // single image SURF
    if(atoi(argv[1]) == 0)
        return mainImage();

    // show match between SURF
    if(atoi(argv[1]) == 1)
        return mainStaticMatch();

    // SURF on webcam
    if(atoi(argv[1]) == 2)
        return mainVideo();

    // SURF with blending
    if(atoi(argv[1]) == 3)
        return mainBlending();

}
