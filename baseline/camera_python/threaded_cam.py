from threading import Thread
import cv2

class CamStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		self.ret_val, self.frame = self.stream.read()

		self.stop_flag = False

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while 1:
			if self.stop_flag:
				break
			self.ret_val, self.frame = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stop_flag = True