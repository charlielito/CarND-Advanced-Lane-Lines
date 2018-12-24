import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
#     return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=2):

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1)) 
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,thickness=2, color=(255,0,0)):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, \
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    left, left_params, right, right_params = get_left_right_lines2(lines, img.shape[1]//2)
    
    if left is not None:
        draw_lines(line_img, left, thickness=thickness, color=color)
        draw_lines(line_img, right, thickness=thickness, color=color)
        
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)

def reject_outliers(data, m=2):
    indexes = abs(data - np.mean(data)) < m * np.std(data)
    return indexes

# Aux function that separates left and right lines
# and calculates its mean slope and b intercept of function y=mx+b
def get_left_right_lines(lines, x_middle):
    left_lines = []
    right_lines = []
    
    left_slope = right_slope = np.array([])
    left_b = right_b = np.array([])
     
    for line in lines:
        
        for x1,y1,x2,y2 in line:
            
            if (x2-x1) != 0: #avoid vertical lines (infinite slope)
                
                slope = (y2-y1)/(x2-x1)
                b = y1 - slope*x1
#                 print(slope,b)
                if slope < -0.4 and x_middle > x2 and x_middle > x1: #must be left lines (filter outlier with slope [-0.1, 0) )
                    left_lines.append(line)
                    left_slope = np.append(left_slope, slope)
                    left_b = np.append(left_b, b)                    
                
                elif slope > 0.4 and x2 > x_middle and x1 > x_middle: #same for right lines
                    right_lines.append(line)
                    right_slope = np.append(right_slope, slope)
                    right_b = np.append(right_b, b)                  
    
    left_index = np.logical_and(reject_outliers(left_b), reject_outliers(left_slope))
    right_index = np.logical_and(reject_outliers(right_b), reject_outliers(right_slope))

    return list(np.array(left_lines)[left_index]), (np.mean(left_slope[left_index]), np.mean(left_b[left_index])), \
           list(np.array(right_lines)[right_index]), (np.mean(right_slope[right_index]), np.mean(right_b[right_index]))


# Aux function that separates left and right lines
# and calculates its mean slope and b intercept of function y=mx+b
def get_left_right_lines2(lines, x_middle):
    left_lines = []
    right_lines = []

    left_lines_x = []
    left_lines_y = []

    right_lines_x = []
    right_lines_y = []
    
    for line in lines:
        
        for x1,y1,x2,y2 in line:
            
            if (x2-x1) != 0: #avoid vertical lines (infinite slope)
                
                slope = (y2-y1)/(x2-x1)

                if slope < -0.4 and x_middle > x2 and x_middle > x1: #must be left lines (filter outlier with slope [-0.1, 0) )
                    left_lines.append(line)
                    left_lines_x.extend([x1,x2])
                    left_lines_y.extend([y1,y2])
        
                elif slope > 0.4 and x2 > x_middle and x1 > x_middle: #same for right lines
                    right_lines.append(line)
                    right_lines_x.extend([x1,x2])
                    right_lines_y.extend([y1,y2])
    if left_lines_x and left_lines_y:
        left_fit = np.polyfit(np.array(left_lines_x), np.array(left_lines_y), 1)
        right_fit  = np.polyfit(np.array(right_lines_x), np.array(right_lines_y), 1)

        return left_lines, left_fit, right_lines, right_fit
    else:
        return None, None, None, None

def fit_lines(lines, y_bottom, y_top, x_middle):
    left, left_params, right, right_params = get_left_right_lines2(lines, x_middle)

    if left is not None:

        yl1 = yr1 = y_bottom
        yl2 = yr2 = y_top # where vertices end

        xl1 = int((yl1 - left_params[1])/left_params[0]) # x = (y-b)/m
        xr1 = int((yr1 - right_params[1])/right_params[0])

        xl2 = int((yl2 - left_params[1])/left_params[0]) # x = (y-b)/m
        xr2 = int((yr2 - right_params[1])/right_params[0])
    
        fitted_lines = [[[xl1, yl1, xl2, yl2]], [[xr1, yr1, xr2, yr2]] ]
        
        return fitted_lines, left, right
    else:
        return None, None, None

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    gray = grayscale(image)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 9
    blur_gray = gaussian_blur(gray,kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    y_top_vertices = int(imshape[0]*0.61)
    vertices = np.array([[(0,imshape[0]),(int(imshape[1]*0.45), y_top_vertices), \
                          (int(imshape[1])*0.55, y_top_vertices), (imshape[1],imshape[0])]], dtype=np.int32)
 
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image, lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, \
                             max_line_gap,thickness=2,color=(0,0,255))

    # Get final fitted right and left lines, and also raw left and right lines for debugging
    fitted_lines, left, right = fit_lines(lines, imshape[0], int(imshape[0]*0.62), imshape[1]//2)


    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on original image
    lines_image = weighted_img(line_image, color_edges, α=0.8, β=1., γ=0.)

    # Draw vertices in image for debugging
    vertices = vertices.reshape((-1,1,2))
    cv2.polylines(lines_image,[vertices],True,(0,255,255))


    final_line_image = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)

    # If could not find any images do not do anything
    if fitted_lines is not None:
        draw_lines(final_line_image, fitted_lines, color=(0,0,255), thickness=10)
        final_image = weighted_img(final_line_image, image, α=0.9, β=1., γ=0.)
    else:
        final_image = image

    return final_image, lines_image, fitted_lines