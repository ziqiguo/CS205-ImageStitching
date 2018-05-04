from panorama_time import Stitcher
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


time_dic = {'featureDetection': [],
			'featureMatching': [],
			'homography': [],
			'warping': [],
			'total': []}

for i in range(200, 4000, 200):
	t0 = time.time()

	# resize for faster processing
	imageA = cv2.imread(args["first"])
	imageB = cv2.imread(args["second"])

	imageA = imutils.resize(imageA, width=i)
	imageB = imutils.resize(imageB, width=i)

	stitcher = Stitcher()
	(result, vis, times) = stitcher.stitch([imageA, imageB], showMatches=True)

	t1 = time.time()
	time_dic['total'].append(t1-t0)
	time_dic['featureDetection'].append(times[0])
	time_dic['featureMatching'].append(times[1])
	time_dic['homography'].append(times[2])
	time_dic['warping'].append(times[3])

print(time_dic)
	# cv2.imshow("Result", result)
	# cv2.imshow("Keypoint Matches", vis)

# cv2.waitKey(0)

# start = time.time()
# imageA = lycon.load(args["first"])
# imageB = lycon.load(args["second"])
# print(time.time() - start, 'seconds')
