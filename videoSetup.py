import cv2
import sys

def init():
	global original
	global augmented
	
	try:
		camera = cv2.VideoCapture("driving.mp4")
	except:
		sys.exit(1)
	ok, original = camera.read()
	augmented = original.copy()
