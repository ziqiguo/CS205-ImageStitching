
#include "integral.h"
#include "ipoint.h"
#include "utils.h"

#include <vector>
#include <ctime>

#include "responselayer.h"
#include "fasthessian.h"

using namespace std;

//-------------------------------------------------------

//! Constructor without image
FastHessian::FastHessian(std::vector<Ipoint> &ipts, const int single_mem_cpy,
                                                 const int octaves, const int intervals, const int init_sample, 
                                                 const float thresh) 
                                                 : ipts(ipts), i_width(0), i_height(0)
{
    // Save parameter set
    saveParameters(octaves, intervals, init_sample, thresh, single_mem_cpy);
}

//-------------------------------------------------------

//! Constructor with image
FastHessian::FastHessian(IplImage *img, std::vector<Ipoint> &ipts, const int single_mem_cpy,
                                                 const int octaves, const int intervals, const int init_sample, 
                                                 const float thresh) 
                                                 : ipts(ipts), i_width(0), i_height(0)
{
    // Save parameter set
    saveParameters(octaves, intervals, init_sample, thresh, single_mem_cpy);

    // Set the current image
    setIntImage(img);
}

//-------------------------------------------------------

FastHessian::~FastHessian()
{
    for (unsigned int i = 0; i < responseMap.size(); ++i)
    {
        delete responseMap[i];
    }
}

//-------------------------------------------------------

//! Save the parameters
void FastHessian::saveParameters(const int octaves, const int intervals, 
                                            const int init_sample, const float thresh, const int single_mem_cpy)
{
    // Initialise variables with bounds-checked values
    this->octaves = 
        (octaves > 0 && octaves <= 4 ? octaves : OCTAVES);
    this->intervals = 
        (intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
    this->init_sample = 
        (init_sample > 0 && init_sample <= 6 ? init_sample : INIT_SAMPLE);
    this->thresh = (thresh >= 0 ? thresh : THRES);

    this->single_mem_cpy = single_mem_cpy;
}


//-------------------------------------------------------

//! Set or re-set the integral image source
void FastHessian::setIntImage(IplImage *img)
{
    // Change the source image
    this->img = img;

    i_height = img->height;
    i_width = img->width;
}

//-------------------------------------------------------

//! Find the image features and write into vector of features
void FastHessian::getIpoints()
{
    // filter index map
    static const int filter_map [OCTAVES][INTERVALS] = {{0,1,2,3}, {1,3,4,5}, {3,5,6,7}, {5,7,8,9}, {7,9,10,11}};

    // Clear the vector of exisiting ipts
    ipts.clear();

    std::clock_t start;
    start = std::clock();

    // Build the response map
    buildResponseMap();

    std::cout << "buildResponseMap took: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    start = std::clock();
    // Get the response layers
    ResponseLayer *b, *m, *t;

    for (int o = 0; o < octaves; ++o) 
    {
        for (int i = 0; i <= 1; ++i)
        {
            b = responseMap.at(filter_map[o][i]);
            m = responseMap.at(filter_map[o][i+1]);
            t = responseMap.at(filter_map[o][i+2]);

            // Ipoint ipts_tmp[t->height][t->width];
            // int changed[t->height][t->width];

            // loop over middle response layer at density of the most 
            // sparse layer (always top), to find maxima across scale and space

            
            // #pragma acc parallel 
            for (int r = 0; r < t->height; ++r)
            {
                for (int c = 0; c < t->width; ++c)
                {
                    if (isExtremum(r, c, t, m, b))
                    {
                        interpolateExtremum(r, c, t, m, b);//, &ipts_tmp[r][c], &changed[r][c]);
                    }
                }
            }

            // for (int r = 0; r < t->height; ++r)
            // {
            //     for (int c = 0; c < t->width; ++c)
            //     {
            //         if (changed[r][c] == 1)
            //         {
            //             ipts.push_back(ipts_tmp[r][c]);
            //         }
            //     }
            // }
        }
    }

    std::cout << "Ipoint vector took: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}

//-------------------------------------------------------

//! Build map of DoH responses
void FastHessian::buildResponseMap()
{
    // Calculate responses for the first 4 octaves:
    // Oct1: 9,    15, 21, 27
    // Oct2: 15, 27, 39, 51
    // Oct3: 27, 51, 75, 99
    // Oct4: 51, 99, 147,195
    // Oct5: 99, 195,291,387

    // Deallocate memory and clear any existing response layers
    for(unsigned int i = 0; i < responseMap.size(); ++i)    
        delete responseMap[i];
    responseMap.clear();

    // Get image attributes
    int w = (i_width / init_sample);
    int h = (i_height / init_sample);
    int s = (init_sample);

    int rsize;

    // Calculate approximated determinant of hessian values
    if (octaves >= 1)
    {
        responseMap.push_back(new ResponseLayer(w,     h,     s,     9));
        responseMap.push_back(new ResponseLayer(w,     h,     s,     15));
        responseMap.push_back(new ResponseLayer(w,     h,     s,     21));
        responseMap.push_back(new ResponseLayer(w,     h,     s,     27));
        rsize = 4;
    }
 
    if (octaves >= 2)
    {
        responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 39));
        responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 51));
        rsize = 6;
    }

    if (octaves >= 3)
    {
        responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 75));
        responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 99));
        rsize = 8;
    }

    if (octaves >= 4)
    {
        responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 147));
        responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 195));
        rsize = 10;
    }

    if (octaves >= 5)
    {
        responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 291));
        responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 387));
        rsize = 12;
    }

    // -------------------single memory copy version-------------------
    if(single_mem_cpy==1)
    {
        float* responses_arr;
        unsigned char* laplacian_arr;
        int step_arr[rsize];
        int b_arr[rsize];
        int l_arr[rsize];
        int w_arr[rsize];
        int height_arr[rsize];
        int width_arr[rsize];

        int total_size = 0;
        int size_arr[rsize];
        size_arr[0] = 0;

        for (unsigned int i = 0; i < responseMap.size(); ++i)
        {
            ResponseLayer* rl = responseMap[i];

            step_arr[i] = rl->step;
            b_arr[i] = (rl->filter - 1) / 2 + 1;
            l_arr[i] = rl->filter / 3;
            w_arr[i] = rl->filter;
            height_arr[i] = rl->height;
            width_arr[i] = rl->width;

            total_size += rl->width*rl->height;
            if(i > 0)
                size_arr[i] = size_arr[i-1] + height_arr[i-1]*width_arr[i-1];
        }

        responses_arr = new float[total_size];
        laplacian_arr = new unsigned char[total_size];

        int img_step = img->widthStep/sizeof(float);
        int img_height = img->height;
        int img_width = img->width;
        float *img_data = (float *) img->imageData;

        // Extract responses from the image
        #pragma acc data copyin(step_arr[0:rsize], b_arr[0:rsize], l_arr[0:rsize], w_arr[0:rsize],\
                 height_arr[0:rsize], width_arr[0:rsize], size_arr[0:rsize], img_step, img_width, img_height, img_data[0:img_height*img_width])\
                  copyout(responses_arr[0:total_size], laplacian_arr[0:total_size])
        {

        #pragma acc parallel loop
        for (unsigned int i = 0; i < responseMap.size(); ++i)
        {

            int l = l_arr[i];
            int w = w_arr[i];
            int b = b_arr[i];
            int step = step_arr[i];

            float Dxx, Dxy, Dyy;

            #pragma acc loop
            for(int ar = 0; ar < height_arr[i]; ++ar) 
            {  
                #pragma acc loop private(Dxx, Dxy, Dyy)
                for(int ac = 0; ac < width_arr[i]; ++ac) 
                {
                    // get the image coordinates
                    int r = ar * step;
                    int c = ac * step; 

                    // Compute response components
                    Dxx = BoxIntegral_acc(img_data, r - l + 1, c - b, 2*l - 1, w, img_step, img_height, img_width)
                            - BoxIntegral_acc(img_data, r - l + 1, c - l / 2, 2*l - 1, l, img_step, img_height, img_width)*3;
                    Dyy = BoxIntegral_acc(img_data, r - b, c - l + 1, w, 2*l - 1, img_step, img_height, img_width)
                            - BoxIntegral_acc(img_data, r - l / 2, c - l + 1, l, 2*l - 1, img_step, img_height, img_width)*3;
                    Dxy = BoxIntegral_acc(img_data, r - l, c + 1, l, l, img_step, img_height, img_width)
                                + BoxIntegral_acc(img_data, r + 1, c - l, l, l, img_step, img_height, img_width)
                                - BoxIntegral_acc(img_data, r - l, c - l, l, l, img_step, img_height, img_width)
                                - BoxIntegral_acc(img_data, r + 1, c + 1, l, l, img_step, img_height, img_width);

                    // Normalise the filter responses with respect to their size
                    Dxx *= 1.f/(w*w);
                    Dyy *= 1.f/(w*w);
                    Dxy *= 1.f/(w*w);

                    (responses_arr+size_arr[i])[ar*width_arr[i] + ac] = (Dxx * Dyy - 0.81 * Dxy * Dxy);
                    (laplacian_arr+size_arr[i])[ar*width_arr[i] + ac] = (Dxx + Dyy >= 0 ? 1 : 0);
                }
            }
        }

        }

        // copy cuda memory back
        int curr_size = 0; 
        for (unsigned int i = 0; i < responseMap.size(); ++i)
        {
            ResponseLayer* rl = responseMap[i];

            memcpy((void*)(rl->responses), (void*)responses_arr+curr_size*sizeof(float), rl->width*rl->height*sizeof(float));
            memcpy((void*)(rl->laplacian), (void*)laplacian_arr+curr_size*sizeof(unsigned char), rl->width*rl->height*sizeof(unsigned char));

            curr_size += rl->width*rl->height;
        }

        delete responses_arr;
        delete laplacian_arr;
    }

    // -------------------multiple memory copy-------------------
    else
    {
        for (unsigned int i = 0; i < responseMap.size(); ++i)
        {
            buildResponseLayer(responseMap[i]);
        }
    }
}


//! Calculate DoH responses for supplied layer
void FastHessian::buildResponseLayer(ResponseLayer *rl)
{
    float *responses = rl->responses;                 // response storage
    unsigned char *laplacian = rl->laplacian; // laplacian sign storage
    int step = rl->step;                                            // step size for this filter
    int b = (rl->filter - 1) / 2 + 1;                 // border for this filter
    int l = rl->filter / 3;                                     // lobe for this filter (filter size / 3)
    int w = rl->filter;                                             // filter size
    float inverse_area = 1.f/(w*w);                     // normalisation factor
    float Dxx, Dyy, Dxy;

    int height = rl->height;
    int width = rl->width;
    int img_step = img->widthStep/sizeof(float);

    int img_height = img->height;
    int img_width = img->width;

    float *img_data = (float *) img->imageData;

    int r, c;

    #pragma acc data copyin(img_data[0:img_height*img_width], height, width, step, Dxx, Dyy, Dxy, l, w, b, \
        inverse_area, img_step, img_height, img_width, r, c) copy(laplacian[0:height*width], responses[0:height*width])
    {
    #pragma acc parallel loop
    for(int ar = 0; ar < height; ++ar) 
    {  
        #pragma acc loop private(r, c, Dxx, Dxy, Dyy)
        for(int ac = 0; ac < width; ++ac) 
        {
            // get the image coordinates
            r = ar * step;
            c = ac * step; 

            // Compute response components
            Dxx = BoxIntegral_acc(img_data, r - l + 1, c - b, 2*l - 1, w, img_step, img_height, img_width)
                    - BoxIntegral_acc(img_data, r - l + 1, c - l / 2, 2*l - 1, l, img_step, img_height, img_width)*3;
            Dyy = BoxIntegral_acc(img_data, r - b, c - l + 1, w, 2*l - 1, img_step, img_height, img_width)
                    - BoxIntegral_acc(img_data, r - l / 2, c - l + 1, l, 2*l - 1, img_step, img_height, img_width)*3;
            Dxy = BoxIntegral_acc(img_data, r - l, c + 1, l, l, img_step, img_height, img_width)
                        + BoxIntegral_acc(img_data, r + 1, c - l, l, l, img_step, img_height, img_width)
                        - BoxIntegral_acc(img_data, r - l, c - l, l, l, img_step, img_height, img_width)
                        - BoxIntegral_acc(img_data, r + 1, c + 1, l, l, img_step, img_height, img_width);

            // Normalise the filter responses with respect to their size
            Dxx *= inverse_area;
            Dyy *= inverse_area;
            Dxy *= inverse_area;

            responses[ar*width + ac] = (Dxx * Dyy - 0.81 * Dxy * Dxy);
            laplacian[ar*width + ac] = (Dxx + Dyy >= 0 ? 1 : 0);
        }
    }

    }

    cout << responses[1] << endl;
}
    
//-------------------------------------------------------

//! Non Maximal Suppression function
int FastHessian::isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
{
    // bounds check
    int layerBorder = (t->filter + 1) / (2 * t->step);
    if (r <= layerBorder || r >= t->height - layerBorder || c <= layerBorder || c >= t->width - layerBorder)
        return 0;

    // check the candidate point in the middle layer is above thresh 
    float candidate = m->getResponse(r, c, t);
    if (candidate < thresh) 
        return 0; 

    int ret_val = 1;
    for (int rr = -1; rr <=1; ++rr)
    {
        for (int cc = -1; cc <=1; ++cc)
        {
            // if any response in 3x3x3 is greater candidate not maximum
            if (
                t->getResponse(r+rr, c+cc) >= candidate ||
                ((rr != 0 || cc != 0) && m->getResponse(r+rr, c+cc, t) >= candidate) ||
                b->getResponse(r+rr, c+cc, t) >= candidate
                ) 
                ret_val = 0;
        }
    }

    return ret_val;
}

//-------------------------------------------------------

//! Interpolate scale-space extrema to subpixel accuracy to form an image feature.     
void FastHessian::interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)//, Ipoint* ipts_tmp, int* changed)
{
    // get the step distance between filters
    // check the middle filter is mid way between top and bottom
    int filterStep = (m->filter - b->filter);
    // assert(filterStep > 0 && t->filter - m->filter == m->filter - b->filter);
 
    // Get the offsets to the actual location of the extremum
    double xi = 0, xr = 0, xc = 0;
    interpolateStep(r, c, t, m, b, &xi, &xr, &xc );

    // If point is sufficiently close to the actual extremum
    if( fabs( xi ) < 0.5f    &&    fabs( xr ) < 0.5f    &&    fabs( xc ) < 0.5f )
    {
        Ipoint ipt;
        ipt.x = static_cast<float>((c + xc) * t->step);
        ipt.y = static_cast<float>((r + xr) * t->step);
        ipt.scale = static_cast<float>((0.1333f) * (m->filter + xi * filterStep));
        ipt.laplacian = static_cast<int>(m->getLaplacian(r,c,t));
        ipts.push_back(ipt);
        // *ipts_tmp = ipt;
        // *changed = 1;
    }
}

//-------------------------------------------------------

//! Performs one step of extremum interpolation. 
void FastHessian::interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b, 
                                                                    double* xi, double* xr, double* xc )
{
    // double** dD, * H, * H_inv, X;
    // double x[3] = {0, 0, 0};

    //--------------------------------------------------------------------------
    // dD = deriv3D( r, c, t, m, b );
    double dI[3][1];
    double dx, dy, ds;

    dx = (m->getResponse(r, c + 1, t) - m->getResponse(r, c - 1, t)) / 2.0;
    dy = (m->getResponse(r + 1, c, t) - m->getResponse(r - 1, c, t)) / 2.0;
    ds = (t->getResponse(r, c) - b->getResponse(r, c, t)) / 2.0;

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 1; j++)
            dI[i][j] = 0;
    dI[0][0] = dx;
    dI[1][0] = dy;
    dI[2][0] = ds;

    //--------------------------------------------------------------------------
    // H = hessian3D( r, c, t, m, b );
    double H[3][3];
    double v, dxx, dyy, dss, dxy, dxs, dys;

    v = m->getResponse(r, c, t);
    dxx = m->getResponse(r, c + 1, t) + m->getResponse(r, c - 1, t) - 2 * v;
    dyy = m->getResponse(r + 1, c, t) + m->getResponse(r - 1, c, t) - 2 * v;
    dss = t->getResponse(r, c) + b->getResponse(r, c, t) - 2 * v;
    dxy = ( m->getResponse(r + 1, c + 1, t) - m->getResponse(r + 1, c - 1, t) - 
                    m->getResponse(r - 1, c + 1, t) + m->getResponse(r - 1, c - 1, t) ) / 4.0;
    dxs = ( t->getResponse(r, c + 1) - t->getResponse(r, c - 1) - 
                    b->getResponse(r, c + 1, t) + b->getResponse(r, c - 1, t) ) / 4.0;
    dys = ( t->getResponse(r + 1, c) - t->getResponse(r - 1, c) - 
                    b->getResponse(r + 1, c, t) + b->getResponse(r - 1, c, t) ) / 4.0;

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            H[i][j] = 0;
    H[0][0] = dxx;
    H[0][1] = dxy;
    H[0][2] = dxs;
    H[1][0] = dxy;
    H[1][1] = dyy;
    H[1][2] = dys;
    H[2][0] = dxs;
    H[2][1] = dys;
    H[2][2] = dss;

    //--------------------------------------------------------------------------
    // H_inv = cvCreateMat( 3, 3, CV_64FC1 );
    // cvInvert( H, H_inv, CV_SVD );
    double det = H[0][0] * (H[1][1] * H[2][2] - H[2][1] * H[1][2]) -
             H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
             H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
    double invdet = 1 / det;

    double H_inv[3][3]; // inverse of matrix H
    H_inv[0][0] = (H[1][1] * H[2][2] - H[2][1] * H[1][2]) * invdet;
    H_inv[0][1] = (H[0][2] * H[2][1] - H[0][1] * H[2][2]) * invdet;
    H_inv[0][2] = (H[0][1] * H[1][2] - H[0][2] * H[1][1]) * invdet;
    H_inv[1][0] = (H[1][2] * H[2][0] - H[1][0] * H[2][2]) * invdet;
    H_inv[1][1] = (H[0][0] * H[2][2] - H[0][2] * H[2][0]) * invdet;
    H_inv[1][2] = (H[1][0] * H[0][2] - H[0][0] * H[1][2]) * invdet;
    H_inv[2][0] = (H[1][0] * H[2][1] - H[2][0] * H[1][1]) * invdet;
    H_inv[2][1] = (H[2][0] * H[0][1] - H[0][0] * H[2][1]) * invdet;
    H_inv[2][2] = (H[0][0] * H[1][1] - H[1][0] * H[0][1]) * invdet;


    //--------------------------------------------------------------------------
    // cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
    // cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );
    double X[3][1] = {{0}, {0}, {0}};

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 1; j++)
        {
            for(int k = 0; k < 3; k++)
                X[i][j] += H_inv[i][k] * dI[k][j];
        }
    }

    //--------------------------------------------------------------------------
    // cvReleaseMat( &dD );
    // cvReleaseMat( &H );
    // cvReleaseMat( &H_inv );

    // *xi = x[2];
    // *xr = x[1];
    // *xc = x[0];
    
    *xi = X[2][0];
    *xr = X[1][0];
    *xc = X[0][0];

}

//-------------------------------------------------------

// //! Computes the partial derivatives in x, y, and scale of a pixel.
// double** FastHessian::deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
// {
//     // CvMat* dI;
//     double** dI;
//     // double dx, dy, ds;

//     // dx = (m->getResponse(r, c + 1, t) - m->getResponse(r, c - 1, t)) / 2.0;
//     // dy = (m->getResponse(r + 1, c, t) - m->getResponse(r - 1, c, t)) / 2.0;
//     // ds = (t->getResponse(r, c) - b->getResponse(r, c, t)) / 2.0;
    
//     // dI = cvCreateMat( 3, 1, CV_64FC1 );
//     // cvmSet( dI, 0, 0, dx );
//     // cvmSet( dI, 1, 0, dy );
//     // cvmSet( dI, 2, 0, ds );
//     // dI = cvCreateMat_rw(3, 1);

//     return dI;
// }

// //-------------------------------------------------------

// //! Computes the 3D Hessian matrix for a pixel.
// double** FastHessian::hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b)
// {
//     double** H;
//     // double v, dxx, dyy, dss, dxy, dxs, dys;

//     // v = m->getResponse(r, c, t);
//     // dxx = m->getResponse(r, c + 1, t) + m->getResponse(r, c - 1, t) - 2 * v;
//     // dyy = m->getResponse(r + 1, c, t) + m->getResponse(r - 1, c, t) - 2 * v;
//     // dss = t->getResponse(r, c) + b->getResponse(r, c, t) - 2 * v;
//     // dxy = ( m->getResponse(r + 1, c + 1, t) - m->getResponse(r + 1, c - 1, t) - 
//     //                 m->getResponse(r - 1, c + 1, t) + m->getResponse(r - 1, c - 1, t) ) / 4.0;
//     // dxs = ( t->getResponse(r, c + 1) - t->getResponse(r, c - 1) - 
//     //                 b->getResponse(r, c + 1, t) + b->getResponse(r, c - 1, t) ) / 4.0;
//     // dys = ( t->getResponse(r + 1, c) - t->getResponse(r - 1, c) - 
//     //                 b->getResponse(r + 1, c, t) + b->getResponse(r - 1, c, t) ) / 4.0;

//     // H = cvCreateMat( 3, 3, CV_64FC1 );
//     // cvmSet( H, 0, 0, dxx );
//     // cvmSet( H, 0, 1, dxy );
//     // cvmSet( H, 0, 2, dxs );
//     // cvmSet( H, 1, 0, dxy );
//     // cvmSet( H, 1, 1, dyy );
//     // cvmSet( H, 1, 2, dys );
//     // cvmSet( H, 2, 0, dxs );
//     // cvmSet( H, 2, 1, dys );
//     // cvmSet( H, 2, 2, dss );

//     return H;
// }

//-------------------------------------------------------
