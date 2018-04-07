import cv2
import sys
import threading
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
import os
import laneDetection
import objectTracking
import NN_ObjectDetection as nn

try:
	camera = cv2.VideoCapture("driving.mp4")
except:
	sys.exit(1)
ok, original = camera.read()

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))
#output = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
#output.open("driving.mp4")

# Uncomment to use the Movidius neural computer stick.
nn.setupMovidius()

# Obsolete use of HOG and Linear SVM for detecting people.
peopleHOG = cv2.HOGDescriptor()
peopleHOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
peopleTracker = objectTracking.TrackedObject(peopleHOG)

# Attempt to use Pipe() and Process().
"""
nn_in, orig_out = Pipe()
fo = Process(target=nn.findObject, args=(nn_in,))
fo.start()
"""

# Attempt at my own implementation of multithreading.

"""
def findAndTrace ():
	ok, frame = camera.read()
	boxes, neural = nn.findObject(frame)
	peopleTracker.updateTraces(frame, boxes)

def do_every (interval, periodic_func):
	periodic_func()
	threading.Timer(interval, periodic_func).start()

do_every(5, findAndTrace)
"""

boxes = nn.findObject(original)
count = 0
period = 30
while camera.isOpened():
	ok, frame = camera.read()

	if ok:
		#orig_out.send_bytes(original, 720) # From attempt to use Pipe()
		lanes = laneDetection.detectLanes(frame)

		if count%period != 0:
			pass
		else:
			#fo.run() # From attempt to use Process()
			boxes = nn.findObject(frame)
			peopleTracker.updateTraces(frame, boxes)

		traces = peopleTracker.getTraces(lanes)
		#boxes = nn.findObject(frame)
		#output.write(boxes)
		cv2.imshow("Output", traces)
		count += 1

	#else:
		#camera.set(cv2.CAP_PROP_POS_FRAMES, 0) # If out of frames, reset the video.

	key = cv2.waitKey(20)
	if key == 27:
		break

camera.release()
#output.release()
exit()
