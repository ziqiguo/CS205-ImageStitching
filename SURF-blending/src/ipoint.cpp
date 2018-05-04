#include <cv.h>
#include <vector>

#include "ipoint.h"
#include "opencv2/calib3d/calib3d.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "utils.h"

// #include "alglib/stdafx.h"
// #include "alglib/alglibmisc.h"

// using namespace alglib;

cv::Mat getBlended(IplImage *img1, IplImage *img2, IpPairVec &matches) 
{   
    cv::detail::MultiBandBlender blender;
    cv::Mat mimg1 = cv::cvarrToMat(img1);
    cv::Mat mimg2 = cv::cvarrToMat(img2);

    cv::Mat H = getHomography(matches, img2);
    cv::Mat mask1(mimg1.size(), CV_8UC1, cvScalar(255));
    // warpPerspective(mask1, mask1, H, mask1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, (0));

    clock_t start = clock();
    cv::Mat warpped;
    warpPerspective(mimg2, warpped, H, cv::Size( mimg2.cols*1.3, mimg2.rows*1), cv::INTER_LINEAR, cv::BORDER_REPLICATE, (0));
    clock_t end = clock();
    std::cout<< "warpping took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;

    // TODO
    start = clock();
    cv::Mat mask2(mimg2.size(), CV_8UC1, cvScalar(255));
    warpPerspective(mask2, mask2, H, cv::Size( mimg2.cols*1.3, mimg2.rows*1), cv::INTER_LINEAR, cv::BORDER_CONSTANT, (0));
    end = clock();
    std::cout<< "creating mask took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;

    start = clock();
    blender.prepare(cv::Rect(0, 0, warpped.cols, warpped.rows));
    //end = clock();
    //std::cout<< "blending0 took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    //start = clock();
    blender.feed(mimg1, mask1, cv::Point2f (0,0));
    //blender.feed(warpped, mask2, translateCorner(H, cv::Point2f (0, 0)));
    //end = clock();
    //std::cout<< "blending1 took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    //start = clock();
    blender.feed(warpped, mask2, cv::Point2f (0, 0));
    //end = clock();
    //std::cout<< "blending2 took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    //start = clock();

    cv::Mat result_s, result_mask;
    blender.blend(result_s, result_mask); //blend
    //end = clock();
    //std::cout<< "blending3 took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    //start = clock();
    result_s.convertTo(result_s, CV_8UC3);
    end = clock();
    std::cout<< "blending took: " << float(end - start) / CLOCKS_PER_SEC << std::endl;
    return result_s;
}

//! Populate IpPairVec with matched ipts 
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches)
{
    float dist, d1, d2;
    Ipoint *match;

    matches.clear();

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
            matches.push_back(std::make_pair(ipts1[i], *match));
        }
    }
}

// void getMatchesKDTree(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches) {

//     Ipoint *match;

//     real_2d_array ipts2_descriptors;
//     ipts2_descriptors.setlength(ipts2.size(), 64);
//     integer_1d_array ipts2_tags;
//     ipts2_tags.setlength(ipts2.size());

//     //parallel omp loop for
//     for (int i = 0; i < ipts2.size(); i++) {
//         ipts2_tags[i] = i;
//         for (int j = 0; j < 64; j++)
//             ipts2_descriptors[i][j] = ipts2[i].descriptor[j];
//     }

//     ae_int_t nx = 64; //dim
//     ae_int_t ny = 0; //optional
//     ae_int_t normtype = 2; //normalize by euclidean dist.
//     kdtree kdt;
//     ae_int_t k;

//     kdtreebuildtagged(ipts2_descriptors, ipts2_tags, nx, ny, normtype, kdt);
//     //omp
//     for (int i = 0; i < ipts1.size(); i++) {
//         real_1d_array query;
//         query.setlength(64);
//         for (int j = 0; j < 64; j++) {
//             query[j] = ipts1[i].descriptor[j];
//         }

//         kdtreequeryknn(kdt, query, 2); //if omp, need to use thread-safe version
//         real_1d_array dists;
//         kdtreequeryresultsdistances(kdt, dists);

//         if (float(dists[0]) / float(dists[1]) < 0.65) {
//             integer_1d_array k;
//             kdtreequeryresultstags(kdt, k);
//             match = &ipts2[int(k[0])];
//             ipts1[i].dx = match->x - ipts1[i].x; 
//             ipts1[i].dy = match->y - ipts1[i].y;
//             matches.push_back(std::make_pair(ipts1[i], *match));
//         }
//     }
// }

cv::Mat getHomography(IpPairVec &matches, IplImage *original) {
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
    std::cout<< "find homography took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
    return H;
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
    std::cout<< "find homography took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // warping took most of the time
    cv::Mat warpped;
    start = clock();
    cv::warpPerspective(moriginal, warpped, H, cv::Size( moriginal.cols*2, moriginal.rows*2));
    end = clock();
    std::cout<< "warpping took: " << float(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    return warpped;
}

cv::Mat getCvStitch(IplImage *src, cv::Mat warpped)
{
    cv::Mat msrc = cv::cvarrToMat(src);

    // cv::Mat mask1(msrc.size(), CV_8UC1, cvScalar(255,255,255));
    // cv::Mat mask2(warpped.size(), CV_8UC1, cvScalar(255,255,255));

    cv::Mat stitched(cv::Size(warpped.cols + msrc.cols,  warpped.rows), CV_8UC3);

    cv::Mat roi1(stitched, cv::Rect(0, 0,  msrc.cols, msrc.rows));
    cv::Mat roi2(stitched, cv::Rect(0, 0, warpped.cols, warpped.rows));

    warpped.copyTo(roi2);
    msrc.copyTo(roi1);

    return stitched;
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

cv::Mat getWarpped(IpPairVec &matches, IplImage *original)//, cv::detail::MultiBandBlender &blender)
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
    cv::Mat warp(h, w*2, CV_8UC3);
    // cv::Mat mask = cv::Mat::zeros(warp.size(), CV_8UC1);


    for(int i = 0; i < h; ++i) {
        for(int j = 0; j < w; ++j) {

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
    //cv::Mat res;
    //blender.blend(res, mask);
    //res.convertTo(res, (res.type() / 8) * 8);
    //cv::imshow("Result", res);
    //cv::Mat smoothed;
    //cv::GaussianBlur(warp, smoothed, cv::Size(3,3), 0.5, 0);
    //cv::medianBlur(warp, smoothed, 5);
    return warp;//smoothed;
}

//
// This function uses homography with CV_RANSAC (OpenCV 1.1)
// Won't compile on most linux distributions
//

//-------------------------------------------------------

//! Find homography between matched points and translate src_corners to dst_corners
cv::Point2f translateCorner(cv::Mat H, const cv::Point src_corners)
{

    //for(int i = 0; i < 4; i++ )
    //{
    double j = src_corners.x, i = src_corners.y;
        double z = 1. / (H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2));     
        double x = (H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2)) * z;
        double y = (H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2)) * z;

        return cv::Point2f(cvRound(x), cvRound(y));
    //}

}


