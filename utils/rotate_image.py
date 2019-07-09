# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="enter the path to image")
args = vars(ap.parse_args())


# load the image from disk
image_path = args["image"]
image = cv2.imread(args["image"]) 
# print(image_path.split('.')[0])i

# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
index = 1
for angle in np.arange(0, 360, 30):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-rot/sequences/uav0000086_00000_v/' + '{:07d}'.format(index) + '.jpg', rotated)
	index = index + 1
	#cv2.imshow("Rotated (Correct)", rotated)
	#cv2.waitKey(0)
