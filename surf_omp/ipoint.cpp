#include <cv.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ipoint.h"

//! Populate IpPairVec with matched ipts 
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches)
{
    float dist, d1, d2;
    Ipoint *match;

    matches.clear();

    unsigned int i, j;

    
    for(unsigned int i = 0; i < ipts1.size(); i++) 
    {
        d1 = d2 = FLT_MAX;

        for(unsigned int j = 0; j < ipts2.size(); j++) 
        {
            dist = ipts1[i] - ipts2[j];    

            if(dist<d1) // if this feature matches better than current best
            {
                d2 = d1;
                d1 = dist;
                match = &ipts2[j];
            }
            else if(dist<d2) // this feature matches better than second best
            {
                d2 = dist;
            }
        }

        // If match has a d1:d2 ratio < 0.65 ipoints are a match
        if(d1/d2 < 0.65) 
        { 
            // Store the change in position
            ipts1[i].dx = match->x - ipts1[i].x; 
            ipts1[i].dy = match->y - ipts1[i].y;
            #pragma omp critical
            matches.push_back(std::make_pair(ipts1[i], *match));
        }
    }
}

//
// This function uses homography with CV_RANSAC (OpenCV 1.1)
// Won't compile on most linux distributions
//

//-------------------------------------------------------

//! Find homography between matched points and translate src_corners to dst_corners
int translateCorners(IpPairVec &matches, const CvPoint src_corners[4], CvPoint dst_corners[4])
{
#ifndef LINUX
    double h[9];
    CvMat _h = cvMat(3, 3, CV_64F, h);
    std::vector<CvPoint2D32f> pt1, pt2;
    CvMat _pt1, _pt2;
    
    int n = (int)matches.size();
    if( n < 4 ) return 0;

    // Set vectors to correct size
    pt1.resize(n);
    pt2.resize(n);

    // Copy Ipoints from match vector into cvPoint vectors
    for(int i = 0; i < n; i++ )
    {
        pt1[i] = cvPoint2D32f(matches[i].second.x, matches[i].second.y);
        pt2[i] = cvPoint2D32f(matches[i].first.x, matches[i].first.y);
    }
    _pt1 = cvMat(1, n, CV_32FC2, &pt1[0] );
    _pt2 = cvMat(1, n, CV_32FC2, &pt2[0] );

    // Find the homography (transformation) between the two sets of points
    if(!cvFindHomography(&_pt1, &_pt2, &_h, CV_RANSAC, 5))    // this line requires opencv 1.1
        return 0;

    // Translate src_corners to dst_corners using homography
    for(int i = 0; i < 4; i++ )
    {
        double x = src_corners[i].x, y = src_corners[i].y;
        double Z = 1./(h[6]*x + h[7]*y + h[8]);
        double X = (h[0]*x + h[1]*y + h[2])*Z;
        double Y = (h[3]*x + h[4]*y + h[5])*Z;
        dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
    }
#endif
    return 1;
}


cv::Mat getWarpped(IplImage *original, cv::Mat H)
{

    cv::Mat src = cv::cvarrToMat(original);
    int h = src.rows, w = src.cols;
    cv::Mat warp(h, w*2, CV_8UC3, cvScalar(0,0,0));
    //cv::Mat mask = cv::Mat::zeros(warp.size(), CV_32SC1);

    int i, j;

    #pragma omp parallel for private(i, j) shared(H, src, warp, h, w)
    for(i = 0; i < h; ++i) {
        for(j = 0; j < w; ++j) {

            double z = 1. / (H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2));
            double x = (H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2)) * z;
            double y = (H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2)) * z;

            if (cvRound(x) >= 0 && cvRound(x) < w*2 && cvRound(y) >= 0 && cvRound(y) < h) {

                cv::Vec3b color = src.at<cv::Vec3b>(cv::Point(j, i));

                if (std::floor(x) != x || std::floor(y) != y) {
                    
                    if (std::floor(x) >= 0 && std::floor(y) >= 0 && std::ceil(x) < w*2 && std::ceil(y) < h ) {

                        warp.at<cv::Vec3b>(cv::Point(std::floor(x), std::floor(y))) = color;
                        warp.at<cv::Vec3b>(cv::Point(std::floor(x), std::ceil(y))) = color;
                        warp.at<cv::Vec3b>(cv::Point(std::ceil(x), std::floor(y))) = color;
                        warp.at<cv::Vec3b>(cv::Point(std::ceil(x), std::ceil(y))) = color;
                            
                    }else{
                        //mask.at<int>(cv::Point(cvRound(x), cvRound(y))) = 1;
                        warp.at<cv::Vec3b>(cv::Point(cvRound(x), cvRound(y))) = color;
                    }
                
                }else{
                    //mask.at<int>(cv::Point(x, y)) = 1;
                    warp.at<cv::Vec3b>(cv::Point(x, y)) = color;
                }
            }
        }
    }

    //cv::Mat smoothed;
    //cv::GaussianBlur(warp, smoothed, cv::Size(3,3), 0.5, 0);
    //cv::medianBlur(warp, smoothed, 5);
    return warp;//smoothed;
}

cv::Mat getCvStitch(IplImage *src, cv::Mat warpped)
{

    cv::Mat msrc = cv::cvarrToMat(src);

    cv::Mat stitched(cv::Size((int)(warpped.cols/2) + msrc.cols,  warpped.rows), CV_8UC3, cvScalar(0,0,0));

    cv::Mat roi1(stitched, cv::Rect(0, 0,  msrc.cols, msrc.rows));
    cv::Mat roi2(stitched, cv::Rect(0, 0, warpped.cols, warpped.rows));

    warpped.copyTo(roi2);
    msrc.copyTo(roi1);

    return stitched;
}

cv::Mat getCvWarpped(IpPairVec &matches, IplImage *original)
{
    std::vector<cv::Point2f> pt1s;
    std::vector<cv::Point2f> pt2s;

    for (int i = 0; i < (int)matches.size(); i++) {
         pt1s.push_back(cv::Point2f(matches[i].second.x, matches[i].second.y));
         pt2s.push_back(cv::Point2f(matches[i].first.x, matches[i].first.y));
    }

    cv::Mat moriginal = cv::cvarrToMat(original);
    clock_t start = clock();
    cv::Mat H = cv::findHomography(pt1s, pt2s, CV_RANSAC);
    clock_t end = clock();
    // std::cout<< "find homography took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // warping took most of the time
    cv::Mat warpped;
    start = clock();
    cv::warpPerspective(moriginal, warpped, H, cv::Size( moriginal.cols*2, moriginal.rows*2));
    end = clock();
    // std::cout<< "warpping took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    return warpped;
}

cv::Mat getWarppedReMap(IpPairVec &matches, IplImage *original)
{
    std::vector<cv::Point2f> pt1s;
    std::vector<cv::Point2f> pt2s;

    for (int i = 0; i < (int)matches.size(); i++) {
        pt1s.push_back(cv::Point2f(matches[i].second.x, matches[i].second.y));
        pt2s.push_back(cv::Point2f(matches[i].first.x, matches[i].first.y));
    }

    cv::Mat H = cv::findHomography(pt1s, pt2s, CV_RANSAC); // 3x3

    cv::Mat src = cv::cvarrToMat(original);
    int h = src.rows, w = src.cols;
    cv::Mat mapX, mapY;
    mapX.create(h, w, CV_32F);
    mapY.create(h, w, CV_32F);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            double z = 1. / (H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2));
            double x = (H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2)) * z;
            double y = (H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2)) * z;
            
            mapX.at<float>(i, j) = x;
            mapY.at<float>(i, j) = y;
        }
    }

    cv::Mat warp;
    cv::remap(src, warp, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return warp;

}

cv::Mat findHom(IpPairVec &matches)
{
    std::vector<cv::Point2f> pt1s;
    std::vector<cv::Point2f> pt2s;

    for (int i = 0; i < (int)matches.size(); i++) {
        pt1s.push_back(cv::Point2f(matches[i].second.x, matches[i].second.y));
        pt2s.push_back(cv::Point2f(matches[i].first.x, matches[i].first.y));
    }

    cv::Mat H = cv::findHomography(pt1s, pt2s, CV_RANSAC); // 3x3
    return H;
}



std::vector<cv::Mat> getWarpped_blend(IplImage *original, cv::Mat H)
{
    double H_[9] = {
        H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), 
        H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), 
        H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2)
    };

    int h = original->height, w = original->width, step = original->widthStep/sizeof(uchar);
    int depth = original->depth, channel = original->nChannels;

    static const double MAX_H = 1.2 * h;

    uchar* src = (uchar*) original->imageData;

    int warpStep = 2 * step;
    int maskStep = w * 2;

    uchar* warp_data = new uchar[h*w*2*channel];
    uchar* mask_data = new uchar[h*w*2];
    memset(warp_data, 0, h*w*2*channel);
    memset(mask_data, 0, h*w*2);

    // int ii, jj, fx, fy, cx, cy;
    // double x, y, z;
    // uchar r, g, b;

    {
    for(int i = (int)(-h/1.2); i < h * 1.2; ++i)
    {
        // int ii = std::min(h-1, std::max(i, 0));
        int ii = i > (h-1) ? h-1 : i<0 ? 0 : i;

        for(int j = (int)(-w/1.2); j < w; ++j)
        {

            double z = 1. / (H_[6]* j + H_[7] * i + H_[8]);
            double x = (H_[0] * j + H_[1] * i + H_[2]) * z;
            double y = (H_[3] * j + H_[4] * i + H_[5]) * z;
       
            // int jj = std::max(j, 0);
            int jj = j>0 ? j : 0;

            uchar b = src[ii*step+jj*channel], g = src[ii*step+jj*channel+1], r = src[ii*step+jj*channel+2];

            if (std::floor(x) >= 0 && std::floor(x) < w*2 && std::floor(y) >= 0 && std::floor(y) < h)
            {

                if (std::floor(x) != x || std::floor(y) != y)
                {

                    int fx = int(std::floor(x)), cx = int(std::ceil(x));
                    int fy = int(std::floor(y)), cy = int(std::ceil(y));
                    
                    if (std::floor(x) >= 0 && std::floor(y) >= 0 && std::ceil(x) < w*2 && std::ceil(y) < h )
                    {

                        warp_data[fy*warpStep + fx*channel] = b;
                        warp_data[fy*warpStep + fx*channel + 1] = g;
                        warp_data[fy*warpStep + fx*channel + 2] = r;

                        warp_data[fy*warpStep + cx*channel] = b;
                        warp_data[fy*warpStep + cx*channel + 1] = g;
                        warp_data[fy*warpStep + cx*channel + 2] = r;

                        warp_data[cy*warpStep + fx*channel] = b;
                        warp_data[cy*warpStep + fx*channel + 1] = g;
                        warp_data[cy*warpStep + fx*channel + 2] = r;

                        warp_data[cy*warpStep + cx*channel] = b;
                        warp_data[cy*warpStep + cx*channel + 1] = g;
                        warp_data[cy*warpStep + cx*channel + 2] = r;

                        if (i >= 0 && i < h && j > 0) {
                            mask_data[fy*maskStep + fx] = 255;
                            mask_data[fy*maskStep + cx] = 255;
                            mask_data[cy*maskStep + fx] = 255;
                            mask_data[cy*maskStep + cx] = 255;
                        }
                            
                    }
                    else
                    {
                        warp_data[fy*warpStep + fx*channel] = b;
                        warp_data[fy*warpStep + fx*channel + 1] = g;
                        warp_data[fy*warpStep + fx*channel + 2] = r;

                        if (i >= 0 && i < h && j > 0)
                        {
                            mask_data[fy*maskStep + fx] = 255;
                        }
                    }
                
                }
                else
                {
                    warp_data[int(y)*warpStep + int(x)*channel] = b;
                    warp_data[int(y)*warpStep + int(x)*channel + 1] = g;
                    warp_data[int(y)*warpStep + int(x)*channel + 2] = r;
                    if (i >= 0 && i < h && j > 0) {
                        mask_data[int(y)*maskStep + int(x)] = 255;
                    }
                }
            }
            // else continue;
        }
    }
    
    }

    cv::Mat warp(h, w*2, CV_8UC3, cvScalar(0, 0, 0));
    cv::Mat mask = cv::Mat::zeros(h, w*2, CV_8UC1);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w*2; j++) {
            warp.at<cv::Vec3b>(cv::Point(j, i)) = cv::Vec3b(warp_data[i*warpStep+j*channel], \
                warp_data[i*warpStep+j*channel + 1], warp_data[i*warpStep+j*channel + 2]);
            mask.at<uchar>(cv::Point(j, i)) = mask_data[i*maskStep+j];
        }
    }
    
    std::vector<cv::Mat> ret;
    ret.push_back(warp);
    ret.push_back(mask);
    return ret;
}


cv::Mat getBlended(IplImage *img1, IplImage *img2, IpPairVec &matches, cv::Mat &warpped, cv::Mat &mask2) 
{   
    cv::detail::MultiBandBlender blender;
    cv::Mat mimg1 = cv::cvarrToMat(img1);
    cv::Mat mimg2 = cv::cvarrToMat(img2);

    cv::Mat mask1(mimg1.size(), CV_8UC1, cvScalar(255));

    
    blender.prepare(cv::Rect(0, 0, warpped.cols, warpped.rows));
    blender.feed(mimg1, mask1, cv::Point2f (0,0));
    blender.feed(warpped, mask2, cv::Point2f (0, 0));
    cv::Mat result_s, result_mask;
    blender.blend(result_s, result_mask); //blend
    result_s.convertTo(result_s, CV_8UC3);

    return result_s;
}

