import cv2
import os

from calibration import get_camera_calibration

# path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

## Get unditort params
mtx, dist = get_camera_calibration(os.path.join(dir_path, "calibration.npz"))

image = cv2.imread("camera_cal/calibration1.jpg")

image_undist = cv2.undistort(image, mtx, dist)

# Resize to fit display
cv2.imshow("Original", cv2.resize(image, (0,0), fx=0.5, fy=0.5))
cv2.imshow("Undistorted", cv2.resize(image_undist, (0,0), fx=0.5, fy=0.5))

while True:
    if cv2.waitKey(30) == ord('q'):
        break