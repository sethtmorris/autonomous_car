import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

def detectObject(orig, hog):
	rects, weights = hog.detectMultiScale(orig, winStride=(4,4), padding=(8,8), scale=1.1)
	
	# Uncomment to not use non-maxima suppression.
	rectlist = []
	for rect in rects:
		rectlist.append(tuple(rect))
	return rectlist

	# Non-maxima suppression. Supposed to remove overlapping detections.
	rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
	picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	picks = np.array([[x1,y1,x2-x1,y2-y1] for (x1,y1,x2,y2) in picks])
	picklist = []
	for pick in picks:
		picklist.append(tuple(pick))

	return picklist
