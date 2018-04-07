import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import NN_ObjectDetection as nn

class TrackedObject(object):
	HOG = cv2.HOGDescriptor()
	Tracker = cv2.TrackerKCF_create()
	Trackers = [Tracker]
	bboxes = []

	def __init__(self, hog):
		self.HOG = hog
		self.bboxes = [(700, 400, 100, 100)]

	def detectObject(self, augmented):
		rects, weights = self.HOG.detectMultiScale(augmented, winStride=(4,4), padding=(8,8), scale=1.1)

		# Uncomment to not use non-maxima suppression.
		rectlist = []
		for rect in rects:
			rectlist.append(tuple(rect))
		self.bboxes = rectlist

		# Non-maxima suppression. Supposed to remove overlapping detections.
		rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
		picks = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		picks = np.array([[x1,y1,x2-x1,y2-y1] for (x1,y1,x2,y2) in picks])
		picklist = []
		for pick in picks:
			picklist.append(tuple(pick))
		self.bbox = picklist

	def updateTraces(self, augmented, boxes):
		tboxes = []
		for box in boxes:
			tboxes.append(tuple(box))

		self.Trackers.clear()
		for tbox in tboxes:
			Tracker = cv2.TrackerKCF_create()
			Tracker.init(augmented, tbox)
			self.Trackers.append(Tracker)

		self.bboxes = tboxes

		return augmented

	#cv2.namedWindow("Traces", cv2.WINDOW_NORMAL)

	def getTraces(self, augmented):
		# Update tracker for all the bounding boxes
		for Tracker in self.Trackers:
			ok, bbox = Tracker.update(augmented)

			# Draw bounding box
			if ok:
				p1 = (int(bbox[1]), int(bbox[0]))
				p2 = (int(bbox[3]), int(bbox[2]))
				cv2.rectangle(augmented, p1, p2, (255,0,0), 2, 1)
			#cv2.imshow("Traces", augmented)
		return augmented

