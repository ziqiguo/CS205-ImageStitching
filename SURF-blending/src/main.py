from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter, maximum_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import cdist
import sys, math, matplotlib.pyplot as plt, numpy as np
import time
import multiprocessing as mp
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

# Converts color image to gray
def rgb2gray(im):
    return 0.299 * im[...,2] + 0.587 * im[..., 1] + 0.114 * im[..., 0]

# ANMS algorithm; Suppressor of points, evenly distributes what points to pick
def anms(coords, top=400):
    l, x, y = [], 0, 0
    while x < len(coords):
        minpoint = 99999999
        xi, yi = coords[x][0], coords[x][1]
        while y < len(coords):   
            xj, yj = coords[y][0], coords[y][1]
            if (xi != xj and yi != yj) and coords[x][2] < 0.9 * coords[y][2]:
                dist = distance(xi, yi, xj, yj)
                if dist < minpoint:
                    minpoint = dist
            y += 1
        l.append([xi, yi, minpoint])
        x += 1
        y = 0
    l.sort(key=lambda x: x[2])
    l = l[0:top]
    print(l)
    return l

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Gets descriptors of an image 
def extract(img, harris, radius=8):
    # Change number here to change the scale. 4 is the optimal amount 
    y, x = 4 * np.mgrid[-radius:radius+1, -radius:radius+1]
    desc = np.zeros((2 * radius + 1, 2 * radius + 1, harris.shape[0]), dtype=float)
    for i in range(harris.shape[0]):
        patch = map_coordinates(img,[harris[i,1] + y, harris[i,0] + x], prefilter=False)
        desc[..., i] = (patch - patch.mean()) / patch.std()
    return desc

# Gets matches of two descriptors 
def matching(d1, d2):
    h, w, n = d1.shape[0:3]
    ds = cdist((d1.reshape((w**2, n))).T, (d2.reshape((w**2, n))).T)
    bt = np.argsort(ds, 1)[:, 0]
    ratio = ds[np.r_[0:n], bt] / ds[np.r_[0:n], np.argsort(ds, 1)[:, 1]].mean()
    return np.hstack([np.argwhere(ratio < 0.5), bt[np.argwhere(ratio < 0.5)]]).astype(int)

# RANSAC algorithm to get best matching pair
def ransac(data, tolerance=0.5, max1=100, confidence=0.95):
    count, bm, bc, bi = 0, None, 0, None
    while count < max1:
        tempd, temps = np.matrix(np.copy(data)), np.copy(data)
        np.random.shuffle(temps) # Gets a random set of points based on RANSAC
        temps = np.matrix(temps)[0:4]
        homography = getHomography(temps[:,0:2], temps[:,2:])
        error = np.sqrt((np.array(np.array(homogeneous((homography * homogeneous(tempd[:,0:2].transpose())))[0:2,:]) - tempd[:,2:].transpose()) ** 2).sum(0))
        if (error < tolerance).sum() > bc:
            bm, bc, bi = homography, (error < tolerance).sum(), np.argwhere(error < tolerance)
            p = float(bc) / data.shape[0]
            max1 = math.log(1-confidence)/math.log(1-(p**4))
        count += 1
    return bm, bi

# Converts 3 x N set of points to homogenous coordinates
def homogeneous(fir):
    if fir.shape[0] == 3:
        out = np.zeros_like(fir)
        for i in range(3):
            out[i, :] = fir[i, :] / fir[2, :]
    elif fir.shape[0] == 2: out = np.vstack((fir, np.ones((1, fir.shape[1]), dtype=fir.dtype)))
    return out

# Harris edge detection
def harris(im, count=512, edge=16):    
    xdir = gaussian_filter1d(gaussian_filter1d(im.astype(np.float32), 1.0, 0, 0), 1.0, 1, 1)
    ydir = gaussian_filter1d(gaussian_filter1d(im.astype(np.float32), 1.0, 1, 0), 1.0, 0, 1)
    h = (gaussian_filter(xdir ** 2, 1.5, 0) * gaussian_filter(ydir ** 2, 1.5, 0) - gaussian_filter(xdir * ydir, 1.5, 0)**2) / (gaussian_filter(xdir**2, 1.5, 0) + gaussian_filter(ydir**2, 1.5, 0) + 1e-8)
    h[:edge, :], h[-edge:, :], h[:, :edge], h[:, -edge:] = 0, 0, 0, 0
    h = h * (h == maximum_filter(h, (8, 8)))
    dirs = np.argsort(h.flatten())[::-1][:count]
    return np.vstack((dirs % im.shape[0:2][1], dirs / im.shape[0:2][1], h.flatten()[dirs])).transpose()

# Gets the homography on an image. Based on 4 point coordinate system.
def getHomography(p1, p2):
    A = np.matrix(np.zeros((p1.shape[0]*2, 8), dtype=float), dtype=float)
    # Filling out A based on equation online
    for i in range(0, A.shape[0]):
        if i % 2 == 0:
            j = int(i/2)
            A[i,0], A[i,1], A[i,2], A[i,6], A[i,7] = p1[j,0], p1[j,1], 1, -p2[j,0] * p1[j,0], -p2[j,0] * p1[j,1]
        else:
            j = int(i/2)
            A[i,3], A[i,4], A[i,5], A[i,6], A[i,7] = p1[j,0], p1[j,1], 1, -p2[j,1] * p1[j,0], -p2[j,1] * p1[j,1]

    # Creating b based on equation
    b = p2.flatten().reshape(p2.flatten().shape[1], 1).astype(float)
    
    # Calculating homography A * x = b
    x = np.linalg.solve(A,b) if p1.shape[0] == 4 else np.linalg.lstsq(A,b)[0]
    return np.vstack((x, np.matrix(1))).reshape((3,3))

# Gets the corners of an image
def corners(homography, files):
    c, mid = [], None
    for i in range(len(files)):
        h, w = cv2.imread(files[i]).shape[0:2]
        c.append(homogeneous(np.dot(homography[i], homogeneous(np.array([[0, w, w, 0], [0, 0, h, h]], dtype=float)))).astype(int))
        if i == len(files)/2:
            mid = c[-1]
    wl, wl2, hl, hl2 = [], [], [], []
    for i in range(len(c)):
        wl, wl2, hl, hl2 = wl + [np.min(c[i][0,:])], wl2 + [np.max(c[i][0,:])], hl + [np.min(c[i][1,:])], hl2 + [np.max(c[i][1,:])]
    return np.array([(min(wl), min(hl)), (max(wl2), max(hl2))]), mid

# Transforms image based on homography
def transformImage(im, t, o="same"):
    yy, xx = np.mgrid[o[0,1]:o[1,1], o[0,0]:o[1,0]]
    h, w = o[1,1] - o[0,1], o[1,0] - o[0,0]   
    i = homogeneous(np.dot(np.linalg.inv(t), homogeneous(np.vstack((xx.flatten(), yy.flatten())))))
    xi, yi = i[0,:].reshape((h, w)), i[1,:].reshape((h, w))
    if im.ndim == 3:
        output = np.zeros((h, w, im.shape[2]), dtype=im.dtype)
        for d in range(im.shape[2]):
            output[..., d] = map_coordinates(im[..., d], [yi, xi])
    else:
        output = map_coordinates(im, [yi, xi])
    return output

# Converts a cv2 image to a numpy array
def nparr(im):
    a = np.fromstring(im.tostring(), dtype='uint8', count=im.shape[0]*im.shape[1]*im.shape[2])
    a.shape = (im.height, im.width, im.nChannels)
    return a

# Displays the feature points created 
def displayPoints(im, arr):
    plt.imshow(im)
    for elem in arr:
        plt.scatter(elem[0], elem[1])
        plt.draw()
    plt.show()

# Main Code
# Getting homography of image files. Supports two for now.
print("Creating the panorama!")
start = time.time()
hh, images, midh, wl = [np.matrix(np.identity(3))], [sys.argv[1], sys.argv[2]], [], []

def load_image(path):
    return cv2.imread(path)

image_loaded = mp.Pool(processes=mp.cpu_count()).map(load_image, images)

def compute_harris(index):
    image = rgb2gray(image_loaded[index])
    points = harris(image, count=500)
    desc = extract(image, points)
    return image, points, desc

t0 = time.time()
for i in range(len(images) - 1):
    
    (image1b, points1, desc1), (image2b, points2, desc2) = mp.Pool(processes=mp.cpu_count()).map(compute_harris, [i,i+1])
    # displayPoints(io.imread(images[i]), points1)
    # modified = anms(points1, top=150)
    # displayPoints(io.imread(images[i]), modified)
    matches = matching(desc1, desc2)
    h = ransac(np.matrix(np.hstack((points1[matches[:,0],0:2], points2[matches[:,1],0:2]))),0.5)
    hh.append(np.linalg.inv(h[0]))
t1=time.time()
print(t1-t0)

# Getting the mid homography from the images provided
hh[1] = hh[0] * hh[1]
for i in range(len(hh)):
    midh.append(np.linalg.inv(hh[int(len(images)/2)]) * hh[i])
t2=time.time()
print(t2-t1)

import pymp


# Warping the images provided to the middle image
for i in range(len(images)):
    im = image_loaded[i]
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    im = np.dstack((im, np.exp(-((yy - im.shape[0]/2) ** 2 + (xx - im.shape[1]/2) ** 2) / (2.0 * 100.0 ** 2)))) # 100.0 is sigma, make higher to blend seams more. 
    wl.append(transformImage(im, np.array(midh[i]), corners(midh, images)[0]))

print(wl[0].shape)
t3=time.time()
print(t3-t2)
# Combining all of the images
t, b = np.zeros(wl[0].shape, dtype=float), np.zeros(wl[0].shape, dtype=float)
b[:,:,3] = 1.0
for i in range(len(wl)):
    t[:,:,0], t[:,:,1], t[:,:,2], t[:,:,3] = t[:,:,0] + wl[i][:,:,3] * wl[i][:,:,0], t[:,:,1] + wl[i][:,:,3] * wl[i][:,:,1], t[:,:,2] + wl[i][:,:,3] * wl[i][:,:,2], t[:,:,3] + wl[i][:,:,3]
    b[:,:,0], b[:,:,1], b[:,:,2] = b[:,:,0] + wl[i][:,:,3], b[:,:,1] + wl[i][:,:,3], b[:,:,2] + wl[i][:,:,3]
b[b == 0] = 1

t4=time.time()
print(t4-t3)
print('Time: ', time.time() - start)
# We are done! Saving! 
cv2.imwrite("image_processed.jpg", t / b)
print("Finished making the panorama, check your source folder!")
