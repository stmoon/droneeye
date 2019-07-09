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

width = image.shape[0]
height = image.shape[1]
cropped_lu = image[0:int(width/2), 0:int(height/2)]
cropped_ru = image[int(width/2):width, 0:int(height/2)]
cropped_ld = image[0:int(width/2), int(height/2):height]
cropped_rd = image[int(width/2):width, int(height/2):height]

cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-split/sequences/uav0000137_00458_v/' + '{:07d}'.format(1) + '.jpg', cropped_lu)
cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-split/sequences/uav0000137_00458_v/' + '{:07d}'.format(2) + '.jpg', cropped_ru)
cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-split/sequences/uav0000137_00458_v/' + '{:07d}'.format(3) + '.jpg', cropped_ld)
cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-split/sequences/uav0000137_00458_v/' + '{:07d}'.format(4) + '.jpg', cropped_rd)
# cv2.imshow("cropped", cropped)
# cv2.waitKey(0)

# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
# index = 1
# for angle in np.arange(0, 360, 30):
# 	rotated = imutils.rotate_bound(image, angle)
# 	cv2.imwrite('/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-split/sequences/uav0000086_00000_v/' + '{:07d}'.format(index) + '.jpg', rotated)
# 	index = index + 1
	#cv2.imshow("Rotated (Correct)", rotated)
	#cv2.waitKey(0)
