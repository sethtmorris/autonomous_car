import cv2
import sys
import multiprocessing
import os
import laneDetection
import objectDetection
import objectTracking

cv2.namedWindow("Original")
cv2.namedWindow("Lanes")
cv2.namedWindow("People")
try:
	camera = cv2.VideoCapture(-1) #"driving.mp4")
except:
	sys.exit(1)

peopleHOG = cv2.HOGDescriptor()
peopleHOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
peopleTracker = objectTracking.TrackedObject(peopleHOG)

carHOG = cv2.HOGDescriptor()

while camera.isOpened():
	ok, original = camera.read()
	if ok:
		#lanes = laneDetection.detectLanes(original)
		newbboxes = objectDetection.detectObject(original, peopleHOG)
		peopleTracker.updateTraces(original, newbboxes)
		people = peopleTracker.getTraces(original)
		#cv2.imshow("Original", original)
		#cv2.imshow("Lanes", lanes)
		cv2.imshow("People", people)

	key = cv2.waitKey(20)
	if key == 27:
		break

cv2.destroyWindow("Original")
cv2.destroyWindow("Lanes")
cv2.destroyWindow("People")
camera.release()
exit()
