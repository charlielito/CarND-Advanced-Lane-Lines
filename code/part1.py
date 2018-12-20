import cv2
import os
import numpy as np
from calibration import get_camera_calibration

# path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_binary_image(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(image)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.dstack(( s_binary, sxbinary, np.zeros_like(sxbinary) )) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    return color_binary, combined_binary



## Undistort test image
mtx, dist = get_camera_calibration(os.path.join(dir_path, "calibration.npz"))

image = cv2.imread("test_images/straight_lines1.jpg")
image_undist = cv2.undistort(image, mtx, dist)

## Create binary image part
color_image, gray_binary = get_binary_image(image_undist, s_thresh=(170,250), sx_thresh=(20,120))

## Visualize
# cv2.imshow("origin", image)
# cv2.imshow("undistorted", image_undist)
# cv2.imshow("Color Binary", color_image)
# cv2.imshow("Binary", gray_binary)
# cv2.waitKey(0)


## Warpped part
offset = 100 # offset for dst points
# Grab the image shape
img_size = (gray_binary.shape[1], gray_binary.shape[0])

src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

print(src)
print(dst)

# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(image_undist, M, img_size)

# Mark images with src and dst
cv2.polylines(image_undist,[np.int32(src)],True,(0,0,255), 2)
cv2.polylines(warped,[np.int32(dst)],True,(0,0,255), 2)


# Visualize
cv2.imshow("Warped area", image_undist)
cv2.imshow("Warpped image", warped)
cv2.waitKey(0)