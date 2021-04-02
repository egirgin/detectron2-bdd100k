import numpy as np
import cv2
import time


class Cam:
    def __init__(self, device_id):
        self.device_id = device_id
        self.cap = None

    def install(self, device_id=None):
        if device_id:
            self.cap = cv2.VideoCapture(device_id)
            self.device_id = device_id
        elif self.device_id != None:
            self.cap = cv2.VideoCapture(self.device_id)
        else:
            print("ERROR!")

    def get_frame(self):
        ret, frame = self.cap.read()

        return ret, frame

    def close_device(self):
        self.cap.release()


flag = 0
my_cap = Cam(0)
my_cap.install()

while (True):

    ret, frame = my_cap.get_frame()
    cv2.imshow('cam0', frame)

    time.sleep(0.25)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

my_cap.close_device()
cv2.destroyAllWindows()
