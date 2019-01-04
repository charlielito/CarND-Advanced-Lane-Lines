import cv2
import os
import numpy as np

from calibration import get_camera_calibration

# path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self): 
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # all good fitted coefficients
        self.all_fit = []   
        # number of consecutive bad frames (not detected)
        self.bad_frames = 0

    # add fit to the list of good fits
    def add_fit(self, fit):
        self.all_fit.append(fit)

    # get the average of the last n fitted coeffs and set it to .best_fit
    def update_best_fit(self, n):
        last_n = np.array(self.all_fit[-n:])
        self.best_fit = np.mean(last_n, axis=0)


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
# offset to better find first line lane (sometimes it identifies something weirds at left side)
def find_line_lane_pixels(binary_warped, histogram, line, offset=0, nwindows=9, margin=100, minpix=50, **kwargs):
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)

    if line == "l":
        x_base = np.argmax(histogram[offset:midpoint]) + offset
        
    else:
        x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    x_current = x_base

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_x_low = x_current - margin  
        win_x_high = x_current + margin 
              
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        # Append these indices to the lists
        lane_inds.append(good_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_inds) > minpix:
            x_current = int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 

    return x, y

def fit_polynomial(binary_warped, left_fit, right_fit, **kwargs):
    
    # Check if there was a previous confident detection
    if left_fit is None or right_fit is None:
    # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        leftx, lefty = find_line_lane_pixels(binary_warped, histogram, "l", offset=100, **kwargs)
        rightx, righty = find_line_lane_pixels(binary_warped, histogram, "r", **kwargs)

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

    else:
        left_fit, left_fitx, ploty, leftx, lefty = search_around_poly(binary_warped, left_fit, margin=50)
        right_fit, right_fitx, ploty, rightx, righty = search_around_poly(binary_warped, right_fit, margin=50)

    ## Visualization ##
    # get only pixels used for regression
    out_imge = np.zeros_like(binary_warped)
    out_imge = np.dstack((out_imge, out_imge, out_imge))
    out_imge[lefty, leftx] = [255, 0, 0]
    out_imge[righty, rightx] = [0, 0, 255]
 
    return out_imge, ploty, left_fit, right_fit, left_fitx, right_fitx


def fit_poly(img_shape, x, y):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    fit = np.polyfit(y, x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2] 
   
    return fit, fitx, ploty

# HYPERPARAMETER
# Choose the width of the margin around the previous polynomial to search
# The quiz grader expects 100 here, but feel free to tune on your own!
def search_around_poly(binary_warped, fit_coeff, margin=60):

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    fit_zeroy_low = fit_coeff[0]*(nonzeroy**2) + fit_coeff[1]*nonzeroy + fit_coeff[2] - margin
    fit_zeroy_high = fit_coeff[0]*(nonzeroy**2) + fit_coeff[1]*nonzeroy + fit_coeff[2] + margin
    lane_inds = (nonzerox > fit_zeroy_low) & (nonzerox < fit_zeroy_high)
    

    # Again, extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 

    # Fit new polynomials
    fit_cr, fitx, ploty = fit_poly(binary_warped.shape, x, y)

    return fit_cr, fitx, ploty, x, y
    

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
    color_image, gray_binary = get_binary_image(image_undist, s_thresh=(130,250), sx_thresh=(20,120))

    # Grab the image shape
    img_size = (gray_binary.shape[1], gray_binary.shape[0])

    ## Now fit a 2nd degree polynomial
    binary_warped = cv2.warpPerspective(gray_binary, M, img_size)

    return binary_warped, M, gray_binary

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

def draw_lane_area(image_undist, binary_warped, Minv, left_fitx, right_fitx, ploty, fitted_pixels_img):
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

    # unwarped fitted pixels
    fit_warp = cv2.warpPerspective(fitted_pixels_img, Minv, (image_undist.shape[1], image_undist.shape[0]))
    # Combine the result with the original image to higlight fitted pixels
    result = cv2.addWeighted(result, 1, fit_warp, 1, 0)

    return result

def pipe_line(image_undist, M, Minv, left_line, right_line):

    binary_warped, M, gray_binary = process_image(image_undist, M)

    fitted_pixels_img, ploty, left_fit_cr, right_fit_cr, left_fitx, right_fitx = \
        fit_polynomial(binary_warped, left_line.best_fit, right_line.best_fit, nwindows=9, margin=50, minpix=25)

    # Curvature
    left_curverad, right_curverad, offset = measure_curvature_offset(
        ploty, 
        left_fit_cr, 
        right_fit_cr, 
        image_undist.shape[1]/2, 
        xm_per_pix, 
        ym_per_pix
    )

    distances = (right_fitx - left_fitx)

    distance = np.mean(distances)*xm_per_pix
    deviation_pix = np.std(distances)

    # lines are not more paralell, keep previous lines
    if deviation_pix > 30 or  not (3.0 < distance < 4.0):
        print("Discarting current detections!")
        left_line.bad_frames += 1

    else:
        left_line.bad_frames = 0
        left_line.add_fit(left_fit_cr)

        right_line.add_fit(right_fit_cr)

        left_line.update_best_fit(10)
        right_line.update_best_fit(10)


    # print(distance, deviation_pix)

    curvature = (left_curverad +  right_curverad)/2

    putText(image_undist, "Radius of Curvature = {}(m)".format(int(curvature)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    position = "left" if offset < 0 else "right"
    putText(image_undist, "Vehicle is {:.2f}(m) {} of center".format(abs(offset), position), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

    #### Draw  lane area
    result = draw_lane_area(image_undist, binary_warped, Minv, left_fitx, right_fitx, ploty, fitted_pixels_img)

    # More than 10 bad detections -> reset en use histogram approach
    if left_line.bad_frames > 10:
        print("Resetting histogram detection!")
        left_line.bad_frames = 0
        left_line.best_fit = None
        right_line.best_fit = None

    return result, left_fit_cr, right_fit_cr, binary_warped, gray_binary


## Save output
SAVE_OUTPUT = False

### Calculate curvature and position to vehicle with respect to center 
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

## Get unditort params
mtx, dist = get_camera_calibration(os.path.join(dir_path, "calibration.npz"))

video_file = 'project_video.mp4'
# video_file = 'challenge_video.mp4'
# video_file = 'harder_challenge_video.mp4'

cap = cv2.VideoCapture(video_file)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 500) #skip first easy part
ret, frame = cap.read()

# Get initial constants required for pipeline
img_size = (frame.shape[1], frame.shape[0])
M = get_M_perspective(img_size)
Minv = np.linalg.inv(M)

left_line = Line()
right_line = Line()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, img_size)
error = 0

while(cap.isOpened()):
    
    if ret:
        image_undist = cv2.undistort(frame, mtx, dist)
        result, left_fit, right_fit, binary_warped, gray_binary = pipe_line(image_undist, M, Minv, left_line, right_line)

        cv2.imshow("Result", result)
        # cv2.imshow("binary", gray_binary)

        if SAVE_OUTPUT:
            out.write(result)

        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cv2.imwrite("test7.jpg", frame)
                break

            while True:
                if cv2.waitKey(1) == ord('c'):
                    break
    
    else:
        error += 1
        if error > 10:
            break

    ret, frame = cap.read()

out.release()
cap.release()
cv2.destroyAllWindows()
