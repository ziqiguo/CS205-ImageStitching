# Real-Time Image Stitching
CS205 Computing Foundations for Computational Science Final Project

Harvard University, 2018 Spring

Team members: Weihang Zhang, Xuefeng Peng, Jiacheng Shi, Ziqi Guo

**This page only provides technical documentations about the project. For a complete view of our project including motivation, design approach, results and analysis, please head over to the [project website](https://cs205-stitching.github.io).** 



## Project Goal

**Image stitching** or photo stitching is the process of combining multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image (example below).

![](images/stitching_example.jpg)

(source: http://www.arcsoft.com/technology/stitching.html)

In this project, we want to use big compute techniques to parallelize the algorithms of image stitching, so that we can stream videos from adjascent camera into a single panoramic view.



## Instructions

### Compile Dependencies:

### Compile:

### Run Test Cases:

### Run:



## References

- SURF code source: https://github.com/julapy/ofxOpenSurf
- Data source: videos taken from [Logitech webcams](https://www.amazon.com/Logitech-Laptop-Webcam-Design-360-Degree/dp/B004YW7WCY/ref=sr_1_8?s=pc&ie=UTF8&qid=1525394553&sr=1-8&keywords=logitech+webcam)
- Knowledge references:
  - Image stitching overview: http://ppwwyyxx.com/2016/How-to-Write-a-Panorama-Stitcher/
  - Keypoint detection: [Speed-Up Robust Features (SURF)](http://www.vision.ee.ethz.ch/~surf/eccv06.pdf)
  - Transformation estimation: [Random Sample Consensus (RANSAC)](http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf)
  - Perspective warping: https://github.com/stheakanath/panorama
  - Blending: Multi-band Blending