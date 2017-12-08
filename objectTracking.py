import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import videoSetup

class TrackedObject(object):
	HOG = cv2.HOGDescriptor()
	TRACKER = cv2.TrackerKCF_create()
	bboxes = []

	def __init__(self, hog):
		self.HOG = hog
		self.bboxes = [(700, 400, 100, 100)]

	def detectObject(self):
		rects, weights = self.HOG.detectMultiScale(augmented, winStride=(4,4), padding=(8,8), scale=1.1)

		# Uncomment to not use non-maxima suppression.
		rectlist = []
		for rect in rects:
			rectlist.append(tuple(rect))
		print("dp")
		#self.bboxes = rectlist

		# Non-maxima suppression. Supposed to remove overlapping detections.
		rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
		picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		picks = np.array([[x1,y1,x2-x1,y2-y1] for (x1,y1,x2,y2) in picks])
		picklist = []
		for pick in picks:
			picklist.append(tuple(pick))
		self.bbox = picklist

	def updateTraces(self, augmented, boxes):
		#self.detectObject()
		for bbox in boxes:
			self.TRACKER.init(augmented, bbox)

		return augmented

	def getTraces(self, augmented, boxes):
		# Update tracker for all the bounding boxes
		for bbox in boxes: #self.bboxes:
			ok, bbox = self.TRACKER.update(augmented)

			# Draw bounding box
			if ok:
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(augmented, p1, p2, (255,0,0), 2, 1)
		cv2.imshow("Traces", augmented)
		return augmented

