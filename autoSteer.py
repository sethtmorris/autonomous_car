import cv2
import sys
import threading
import os
import videoSetup
import laneDetection
import objectTracking
import NN_ObjectDetection as nn

cv2.namedWindow("Original")
cv2.namedWindow("Augmented")

videoSetup.init()

try:
	camera = cv2.VideoCapture("driving.mp4")
except:
	sys.exit(1)
ok, original = camera.read()
augmented = original.copy()

nn.setupMovidius()

peopleHOG = cv2.HOGDescriptor()
peopleHOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
peopleTracker = objectTracking.TrackedObject(peopleHOG)

carHOG = cv2.HOGDescriptor()

def do_every (interval, periodic_func):
	threading.Timer(interval, do_every, [interval, periodic_func]).start()
	return periodic_func()

#do_every(1, peopleTracker.updateTraces)
do_every(0.5, laneDetection.detectLanes)
#do_every(0.5, peopleTracker.getTraces)

while camera.isOpened():
	ok, original = camera.read()
	if ok:
		augmented = original.copy()
		neural = nn.findObject(augmented)
		cv2.imshow("Original", original)
		cv2.imshow("Augmented", neural)

	key = cv2.waitKey(20)
	if key == 27:
		break

cv2.destroyWindow("Original")
cv2.destroyWindow("Augmented")
camera.release()
exit()
