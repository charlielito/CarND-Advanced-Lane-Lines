import cv2
import os
import numpy as np

from calibration import get_camera_calibration

from project1 import process_image
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

# Choose the number of sliding windows
# Set the width of the windows +/- margin
# Set minimum number of pixels found to recenter window
offset = 100
def find_lane_pixels(binary_warped, nwindows=9, margin=50, minpix=25, **kwargs):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[offset:midpoint]) + offset
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin 
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds):
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, **kwargs):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, **kwargs)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # get only pixels used for regression
    out_imge2 = np.zeros_like(out_img)
    out_imge2[lefty, leftx] = [255, 0, 0]
    out_imge2[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    draw_poits(out_img, left_fitx, ploty)
    draw_poits(out_img, right_fitx, ploty)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, ploty, left_fit, right_fit, left_fitx, right_fitx, out_imge2

def draw_poits(image, xs,ys):
    color = (0,255,255)
    thickness = 1
    for x,y in zip(xs,ys):
        cv2.circle(image, (int(x),int(y)), 1, color, thickness)

def measure_curvature_offset(ploty, left_fit_cr, right_fit_cr, image_center, xm_per_pix, ym_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''   
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate offset of center of the lane (asume center of image is center of car)
    left_fitx = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
    right_fitx = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
    center_lane = left_fitx + (right_fitx - left_fitx)/2

    real_offset = (image_center - center_lane)*xm_per_pix

    return left_curverad, right_curverad, real_offset

def putText(img, msg, pos, font, size, color, thickness):
    cv2.putText(img, msg, pos, font, size, (0,0,0), thickness+5)
    cv2.putText(img, msg, pos, font, size, color, thickness)


## Undistort test image
mtx, dist = get_camera_calibration(os.path.join(dir_path, "calibration.npz"))

image = cv2.imread("test_images/straight_lines1.jpg")
# image = cv2.imread("test_images/straight_lines2.jpg")
# image = cv2.imread("test_images/test1.jpg")
# image = cv2.imread("test_images/test2.jpg")
# image = cv2.imread("test_images/test3.jpg")
# image = cv2.imread("test_images/test4.jpg")
# image = cv2.imread("test_images/test5.jpg")
# image = cv2.imread("test_images/test6.jpg")
# image = cv2.imread("test_images/test7.jpg")


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

final_image, lines_image, fitted_lines = process_image(image_undist)

# process list and adjust to proper format for scr points
# src = np.array(fitted_lines, dtype=np.float32).reshape(4,2)[[0,1,3,2]]
# src[0][0] -= 15 #manual adjust for better parallel lines in later warped part
# src[1][0] -= 4 #manual adjust for better parallel lines in later warped part
# src[2][0] += 4 #manual adjust for better parallel lines in later warped part
# src[3][0] += 15 #manual adjust for better parallel lines in later warped part

src = np.float32([[ 190.0,  720.0],
 [ 599.0,  446.0],
 [ 681.0, 446.0],
 [1122.0,  720.0]]
)

dst = np.float32(
    [[(img_size[0] / 4), img_size[1]],
    [(img_size[0] / 4), 0],
    [(img_size[0] * 3 / 4), 0],
    [(img_size[0] * 3 / 4), img_size[1]]])

print(src)
print(dst)

# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(image_undist, M, img_size)

# Mark images with src and dst
image_undist_cp = image_undist.copy()
cv2.polylines(image_undist_cp,[np.int32(src)],True,(0,0,255), 2)
cv2.polylines(warped,[np.int32(dst)],True,(0,0,255), 2)

# Visualize
cv2.imshow("Warped area", image_undist_cp)
cv2.imshow("Warpped image", warped)
# cv2.imshow("Project1", final_image)
cv2.waitKey(0)

## Now fit a 2nd degree polynomial
binary_warped = cv2.warpPerspective(gray_binary, M, img_size)

# leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
out_img, ploty, left_fit_cr, right_fit_cr, left_fitx, right_fitx, out_img2 = fit_polynomial(binary_warped)
cv2.imshow("Gray Warped", binary_warped)
cv2.imshow("Process", out_img)
cv2.waitKey(0)

### Calculate curvature and position to vehicle with respect to center 
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Curvature
left_curverad, right_curverad, offset = measure_curvature_offset(
    ploty, 
    left_fit_cr, 
    right_fit_cr, 
    img_size[0]/2, 
    xm_per_pix, 
    ym_per_pix
)
print(type(right_fit_cr))
print(left_curverad, 'm', right_curverad, 'm', offset, 'm')
curvature = (left_curverad +  right_curverad)/2

putText(image_undist, "Radius of Curvature = {}(m)".format(int(curvature)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
position = "left" if offset < 0 else "right"
putText(image_undist, "Vehicle is {:.2f}(m) {} of center".format(abs(offset), position), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

#### Draw  lane area

# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

Minv = np.linalg.inv(M)
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image_undist.shape[1], image_undist.shape[0]))

# Warp pixels used for regression
fit_warp = cv2.warpPerspective(out_img2, Minv, (image_undist.shape[1], image_undist.shape[0]))

# Combine the result with the original image
result = cv2.addWeighted(image_undist, 1, newwarp, 0.3, 0)
result = cv2.addWeighted(result, 1, fit_warp, 1, 0)
cv2.imshow("Final", result)
cv2.waitKey(0)


