import cv2
import os
import numpy as np

from calibration import get_camera_calibration

# path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # average x values of the fitted line over the last n iterations
        self.bestx = None     
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None  


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
def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50, **kwargs):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
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

    # Plots the left and right polynomials on the lane lines
    draw_poits(out_img, left_fitx, ploty)
    draw_poits(out_img, right_fitx, ploty)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, ploty, left_fit, right_fit, left_fitx, right_fitx

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


def process_image(image_undist, M):

    ## Create binary image part
    color_image, gray_binary = get_binary_image(image_undist, s_thresh=(170,250), sx_thresh=(20,120))

    ## Visualize
    # cv2.imshow("origin", image)
    # cv2.imshow("undistorted", image_undist)
    # cv2.imshow("Color Binary", color_image)
    # cv2.imshow("Binary", gray_binary)
    # cv2.waitKey(0)

    ## Warpped part
    # Grab the image shape
    img_size = (gray_binary.shape[1], gray_binary.shape[0])

    # Warp the image using OpenCV warpPerspective()
    # warped = cv2.warpPerspective(image_undist, M, img_size)

    # # Mark images with src and dst
    # image_undist_cp = image_undist.copy()
    # cv2.polylines(image_undist_cp,[np.int32(src)],True,(0,0,255), 2)
    # cv2.polylines(warped,[np.int32(dst)],True,(0,0,255), 2)

    # Visualize
    # cv2.imshow("Warped area", image_undist_cp)
    # cv2.imshow("Warpped image", warped)
    # # cv2.imshow("Project1", final_image)
    # cv2.waitKey(0)

    ## Now fit a 2nd degree polynomial
    binary_warped = cv2.warpPerspective(gray_binary, M, img_size)

    return binary_warped, M

def get_M_perspective(img_size):

    src = np.float32(
        [[ 190.0,  720.0],
        [ 599.0,  446.0],
        [ 681.0, 446.0],
        [1122.0,  720.0]]
    )

    dst = np.float32(
        [[(img_size[0] / 4), img_size[1]],
        [(img_size[0] / 4), 0],
        [(img_size[0] * 3 / 4), 0],
        [(img_size[0] * 3 / 4), img_size[1]]]
    )

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    return M

def draw_lane_area(image_undist, binary_warped, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_undist.shape[1], image_undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image_undist, 1, newwarp, 0.3, 0)

    return result

def pipe_line(image_undist, M, Minv):

    binary_warped, M = process_image(image_undist, M)

    out_img, ploty, left_fit_cr, right_fit_cr, left_fitx, right_fitx = fit_polynomial(binary_warped)

    # Curvature
    left_curverad, right_curverad, offset = measure_curvature_offset(
        ploty, 
        left_fit_cr, 
        right_fit_cr, 
        image_undist.shape[1]/2, 
        xm_per_pix, 
        ym_per_pix
    )

    curvature = (left_curverad +  right_curverad)/2

    putText(image_undist, "Radius of Curvature = {}(m)".format(int(curvature)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    position = "left" if offset < 0 else "right"
    putText(image_undist, "Vehicle is {:.2f}(m) {} of center".format(abs(offset), position), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

    #### Draw  lane area
    result = draw_lane_area(image_undist, binary_warped, Minv, left_fitx, right_fitx, ploty)

    return result

### Calculate curvature and position to vehicle with respect to center 
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

## Get unditort params
mtx, dist = get_camera_calibration(os.path.join(dir_path, "calibration.npz"))

video_file = 'project_video.mp4'

cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()

# Get initial constants required for pipeline
img_size = (frame.shape[1], frame.shape[0])
M = get_M_perspective(img_size)
Minv = np.linalg.inv(M)

while(cap.isOpened()):
    
    if ret:
        image_undist = cv2.undistort(frame, mtx, dist)
        result = pipe_line(image_undist, M, Minv)

        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        while True:
            if cv2.waitKey(33) == ord('c'):
                break
    
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
