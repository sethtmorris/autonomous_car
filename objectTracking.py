import cv2

class TrackedObject(object):
	HOG = cv2.HOGDescriptor()
	TRACKER = cv2.TrackerKCF_create()
	bboxes = []

	def __init__(self, hog):
		self.HOG = hog
		self.bboxes = [(700, 400, 100, 100)]

	def updateTraces(self, orig, newbboxes):
		self.bboxes = newbboxes
		for bbox in self.bboxes:
			#print(bbox)
			self.TRACKER.init(orig, bbox)

	def getTraces(self, orig):
		trace = orig.copy()

		# Update tracker for all the bounding boxes
		for bbox in self.bboxes:
			ok, bbox = self.TRACKER.update(trace)

			# Draw bounding box
			if ok:
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(trace, p1, p2, (255,0,0), 2, 1)

		return trace

