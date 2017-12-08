import cv2
import sys
import threading
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
import os
import laneDetection
import objectTracking
import NN_ObjectDetection as nn

#cv2.namedWindow("Original")
#cv2.namedWindow("Augmented")

try:
	camera = cv2.VideoCapture("driving.mp4")
except:
	sys.exit(1)
ok, original = camera.read()

nn.setupMovidius()

peopleHOG = cv2.HOGDescriptor()
peopleHOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
peopleTracker = objectTracking.TrackedObject(peopleHOG)

#nn_imagepipe = Queue(30)
#nn_in, orig_out = Pipe()
#fo = Process(target=nn.findObject, args=(nn_imagepipe,))
#fo.start()

def do_every (interval, periodic_func):
	ok, frame = camera.read()
	periodic_func(frame)
	threading.Timer(interval, periodic_func).start()

#do_every(1, peopleTracker.updateTraces)
#do_every(0.5, laneDetection.detectLanes)
#do_every(0.5, peopleTracker.getTraces)
#do_every(1, nn.findObject)

boxes = []
count = 0
period = 30
while camera.isOpened():
	ok, original = camera.read()

	if ok:
		#augmented = original.copy()
		#orig_out.send_bytes(original, 720)
		lanes = laneDetection.detectLanes(original)
		#nn_imagepipe.put(original)		
		if count%period != 0:
			pass
		else:
			boxes, neural = nn.findObject(lanes)
			#fo.run()
		#peopleTracker.updateTraces(original, boxes)
		#peopleTracker.getTraces(original, boxes)
		count += 1
		#cv2.imshow("Original", original)
		#cv2.imshow("Augmented", augmented)

	else:
		camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

	key = cv2.waitKey(20)
	if key == 27:
		break

#cv2.destroyWindow("Original")
#cv2.destroyWindow("Augmented")
camera.release()
exit()
