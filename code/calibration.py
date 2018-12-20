import cv2
import numpy as np
import glob
import sys
import os

# path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Function that takes a list of BGR images, the paremeters of 
# the chessboard pattern (nx, ny) and returns camera matrix and distortion
def calibrate_camera(images_list, nx, ny, show_images=True):

    # Camera calibration
    objpoints = [] #3D point in real world space
    imgpoint = [] #2D points in image plane

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for image in images_list:

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret: #found chessboard
            imgpoint.append(corners)
            objpoints.append(objp)

            # Optional show each image 200 ms
            if show_images:
                image = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
                cv2.imshow("FOUND PATTERN", image)
                cv2.waitKey(200)
        else:
            if show_images:
                cv2.imshow("NOT FOUND", image)
                cv2.waitKey(200)
    
    # If no more than 5 patterns found do not calculate it
    if len(imgpoint) > 5:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoint, gray.shape[::-1], None, None)
        return mtx, dist
    else:
        print("Too few pattern found!")
        return None, None

# Aux function to read the camera matrix and distortion from a given path with the .npz file
def get_camera_calibration(path_file):
    npzfile = np.load(path_file)
    return npzfile['arr_0'], npzfile['arr_1']


# Main script
if __name__ == '__main__':

    images = glob.glob(os.path.join(dir_path,"../camera_cal/calibration*.jpg"))

    # Read all images into a list
    cv_images = list(map(cv2.imread, images))

    # Calculate parameters
    mtx, dist = calibrate_camera(cv_images, 9, 6, show_images=True)

    # Save parameters if found
    if mtx is not None:
        np.savez(os.path.join(dir_path,"calibration.npz"), mtx, dist)
