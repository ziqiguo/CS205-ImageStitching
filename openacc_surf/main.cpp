#include "surflib.h"
// #include "kmeans.h"
#include <ctime>
#include <iostream>


int mainImage(int single_mem_cpy)
{
    // Declare Ipoints and other stuff
    IpVec ipts;
    // IplImage *img=cvLoadImage("../imgs/IMG_5053.jpg");
    IplImage *img=cvLoadImage("../images/img1.jpg");
    

    // Detect and describe interest points in the image
    clock_t start = clock();
    surfDetDes(img, ipts, single_mem_cpy, false, 1, 4, 2, 0.0004f); 
    clock_t end = clock();

    std::cout<< "Found: " << ipts.size() << " interest points" << std::endl;
    std::cout<< "Took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds" << std::endl;

    // Draw the detected points
    drawIpoints(img, ipts);
    
    // Display the result
    showImage(img);

    return 0;
}

//-------------------------------------------------------

int mainVideo(int single_mem_cpy)
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
        surfDetDes(img, ipts, single_mem_cpy, false, 4, 4, 2, 0.004f);        

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


int mainStaticMatch(int single_mem_cpy)
{
    IplImage *img1, *img2;
    img1 = cvLoadImage("../images/img1.jpg");
    img2 = cvLoadImage("../images/img2.jpg");

    clock_t start = clock();
    IpVec ipts1, ipts2;
    surfDetDes(img1,ipts1,single_mem_cpy,false,4,4,2,0.0001f);
    surfDetDes(img2,ipts2,single_mem_cpy,false,4,4,2,0.0001f);
    clock_t end = clock();
    std::cout<< "Took: " << float(end - start) / CLOCKS_PER_SEC    << " seconds to find featurs in 2 pics" << std::endl;


    IpPairVec matches;
    getMatches(ipts1,ipts2,matches);

    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        drawPoint(img1,matches[i].first);
        drawPoint(img2,matches[i].second);
    
        const int & w = img1->width;
        cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
        cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
    }

    std::cout<< "Matches: " << matches.size();

    cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
    cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
    cvShowImage("1", img1);
    cvShowImage("2",img2);
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
        return mainImage(atoi(argv[2]));

    // show match between SURF
    if(atoi(argv[1]) == 1)
        return mainStaticMatch(atoi(argv[2]));

    // SURF on webcam
    if(atoi(argv[1]) == 2)
        return mainVideo(atoi(argv[2]));

}
