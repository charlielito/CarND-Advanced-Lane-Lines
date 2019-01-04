## Advanced Lane Finding Project

**Carlos Andres Alvarez, Udacity Self Driving Nanodegree**


[//]: # (Image References)

[image1]: ./output_images/cal_check.png "Undistorted"
[image2a]: ./test_images/straight_lines1.jpg "Road Transformed"
[image2b]: ./output_images/undistorted.jpg "Road UndisTransformed"
[image3a]: ./output_images/gray_binary.jpg "Binary Example"
[image3b]: ./output_images/color_binary.jpg "Binary Example2"
[image4]: ./output_images/srcwarped.png "Warp Example"
[image5a]: ./output_images/histogram.png "Histogram process"
[image5b]: ./output_images/fitted_lines.jpg "Fit Visual"
[image6]: ./output_images/final.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[videogif]: https://raw.githubusercontent.com/charlielito/CarND-Advanced-Lane-Lines/master/project_video_result.gif "S"

---

### Camera Calibration

#### Computation of camera matrix and distortion coefficients

The code for this step is contained in the script located in `./code/calibration.py`. It has a function called `calibrate_camera` that receives a list of images to base the calibration on and the `nx` and `ny` of the chessboard and finds the calibration parameters.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. There is a flag in the function  `show_images` to show each image and the chessboard detection one by one for debugging.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function if and only if there are more than 5 pattern detected in the image dataset. This because with less images the calibration is not that good. At the end the function returns as a tuple the camera matrix and the distortion coefficients, or a tuple of `None` if the last check was not successful.

I used that function to get the parameters and if found, save them as a `.npz` in the `code` folder. I applied this distortion correction to the test image using the `cv2.undistort()` function after reading the `calibration.npz` file with the `get_camera_calibration` function and obtained this result (running the script `code/calibration_test.py`):

![alt text][image1]

### Pipeline (single images)

This pipeline is in the script `code/base_line.py`

#### 1. Distortion-corrected image test

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
**Original**
![alt text][image2a]
**Undistorted**
![alt text][image2b]


#### 2. Create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image: function `get_binary_image` in the script mentioned at the beginning of this section. I used the HLS space for color thresholding since in this space the yellow lines and white lines can be detected using the S channel without much problem. I also used the Sobel transform in the `x` direction since this direction is good for detecting vertical lines. This function receives the thresholding limits as a tuple, for S channel and `x` Sobel absolute values thresholding. The values used are `(130,250)` and `(20,120)` respectively as you can see in line 215 of `base_line.py`.

Here's an example of my output for this step

![alt text][image3a]

And here you can see the contribution of each thresholding technique to the previous image: blue are the pixels from S threshold and green from Sobel threshold.

![alt text][image3b]


#### 3. Perspective transform

For this part I re-used the code of my [first projecct](https://github.com/charlielito/CarND-LaneLines-P1) of the nanodegree. Y used the code to get the beginning and end of the left and right lines for one of the images in the `test_images` folder to get an approximate of the source `src` points. That first project is copied in the file `code/project1.py` from where the function `process_image` is imported, that returns the lines points of interest -> line 232 in `code/base_line.py`. After that I tweaked a little the points (lines 236-239) to be not exactly at the lines, but just at the border of the lines to have the following result having the lines appear parallel in the warped image:

![alt text][image4]

After having that defined, I commented those lines for the other images and hardcoded the `src` and `dst` points as follows:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 190, 720      | 320, 720      |
| 599, 446      | 320, 0        |
| 681, 446      | 960, 0        |
| 1122, 720     | 960, 720      |

The code for my perspective transform appears in lines 256 - 259

```python
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(image_undist, M, img_size)
```

#### 4. Identify lane-line pixels and fit their positions with a polynomial

This part is done by the function `fit_polynomial` that receives the binary warped image and returns the fitted polynomials coefficients, the `y` and `x` pixel positions of the fitted polynomial and some images for debugging and visualization with the processes. Nonetheless, the main work is done inside by a function called `find_lane_pixels`. This function returns the pixels that are going to be used for the regressions.

The latter is done by the histogram and sliding window approach. Using the histogram we find the two starting positions of the lines as you can see in the next image. Because sometimes the highway separating wall is recognized as a line, this histogram search starts with an offset of 100 pixels to the right. After that we run a sliding window up to recenter the window if more than a threshold of pixels are found. I used 9 sliding windows, with a margin of 75 pixels and a minimum of pixels of 25.

![alt text][image5a]

This sliding windows positions can be seen in the next image. Also after having those pixels, I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5b]


#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

For this I assume what was indicated in the course to calculate the relation meter/pixel. So for projecting a section of lane similar to the images above, it was assumed that the lane is about 30 meters long and 3.7 meters wide. So in the warped space the pixels between lines are approximately 700 and the height is 720. So that translates to a relation of:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

Then, the actual curvature is calculated using a function called `measure_curvature_offset` that receives the values of `y` of the lines, right and left fits, the center of image and and the `x` and `y` meter/pixel relation, and return the vehicle position with respect to center and the curvatures of the left and right lines all in meters

Given that the fitted polynomial has the form:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f(y)=Ay^2+By+c" title="eq1" />

We get from Calculus that the curvature at a point `y` is:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;R(x)=\frac{(1+(2Ay+B)^2)^{3/2}}{\mid 2A \mid}" title="eq2" />

So in that function, this formula is applied taking into account that the value of `y` is previously multiplied by the `ym_per_pix` value to get the curvature in meters. The value of `y` is taken to be the bottom of the image. To calculate the position of the vehicle with respect to center I just calculated the center of the to fitted lines at the bottom of the image, and the extracted to the actual center of the image and multiplied that value by `xm_per_pix` to get how many meter the vehicle is deviated from center. Finally just an average of the two curvatures is displayed in the final result image


#### 6. Result plotted back down onto the road

I implemented this step in lines 305 through 325 in my code in `code/base_line.py`. Basically it creates an image with a filled polygon between the fitted lines, unwarpes that image to the original space and then it is `addWeighted` to the original image. Also to visualize the pixels used for the quadratic regression an image with those pixels (in blue for the left and red for the right lines) is `addWeighted` with the original. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

This pipeline to run on videos is implemented in `code/pipeline.py` and most of the functions are based on the previous sections. It just wrap up some of the code in functions in order to work with a video or multiple consecutive images.

#### 1. Modifications to work with video

To track of things like where the last several detections of the lane lines were and what the curvature was, it's useful to define a Line() class to keep track of all the interesting parameter, so new detections can properly treated.

This class looks like:

```python
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
```

It tracks all good fit coefficients, and also has a counter of bad frames when the lines are not detected properly. It also updates the best fit by averaging the last `n` best coefficients to smooth out the behaviour of the algorithm.

Another function was implemented called `search_around_poly` that skips the histogram part, and searches pixels for the next regression around the previous detected polynomial given a margin in pixels. I used 50 pixels as margin.

The magic occurs in the function `pipeline`. It receives the 2 lines objects to keep track of past detections and the current image. The procedure is the same explained as in the previous sections. Just that sanity checks are made for each detection. It calculates the `x` distances in meters between each point of the fitted lines to check if the distance is correct according to the norms, i.e. approx 3.7 meters. So it checks if the average of those distances fall between 3 and 4 meters. It also checks if the standard deviation in pixels is too high (meaning the lines are not that parallel). If any of this conditions hold, the current detection is rejected and the previous detection preserved. If this happens 10 times consecutively, the histogram approach is used to get the next estimation of the lane.


#### 2. Final video output.

Here's a [link to my video result (full HD)](./output.mp4)

Or just see the video gif at 3x speed (low res)

![alt text][videogif]

---

### Discussion

#### 1. Problems / issues in the implementation of this project. Failures? More robust?

I used a hardcoded `src` region for the warped part. This was little difficult to tweak for the lines that were parallel, to actually be also parallel in the warped space. I think this could fail for other videos or other conditions since those numbers are fixed. I think it could be improved by defining it better like as a function of the image size.

The techniques for getting the binary image where HSL thresholding and Sobel gradients. This works for the video of this project, but I had to tweak the parameters also to work where the highway was lighter. I think this could fail for other light or road conditions. Maybe using other techniques combined could generate a more robust solution, or training some deep learning network for predicting the pixels lanes.

I also noticed that the algorithm does not run as fast as expected, maybe the quadratic regression is a heavy task. Maybe using series of Hough transforms could speed up the curve estimation. Also the meter/pixel relation were defined assuming a lot of things. Having other way to measure that relation could improve the radius estimation and position of the car.
