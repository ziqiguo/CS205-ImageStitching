from panorama import Stitcher
import argparse
import imutils
import cv2
import time
#import lycon

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
ap.add_argument("-w", "--width", required=True,
	help="image width for stitching")
args = vars(ap.parse_args())


t0 = time.time()

# resize for faster processing
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

if args['width'] != 'None':
	imageA = imutils.resize(imageA, width=int(args["width"]))
	imageB = imutils.resize(imageB, width=int(args["width"]))

print(imageA.shape)
 
t1 = time.time()
print('Image loading:', t1-t0)

stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

t2 = time.time()
print('Finish stitching:', t2-t1)

cv2.imshow("Result", result)

t3 = time.time()
print('Display image:', t3-t2)
 
print('Total time:', time.time() - t0, 'seconds')

cv2.imshow("Keypoint Matches", vis)

# cv2.waitKey(0)

# start = time.time()
# imageA = lycon.load(args["first"])
# imageB = lycon.load(args["second"])
# print(time.time() - start, 'seconds')
