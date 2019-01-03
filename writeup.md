## Advanced Lane Finding Project

**Carlos Andres Alvarez, Udacity Self Driving Nanodegree**


[//]: # (Image References)

[image1]: ./output_images/cal_check.png "Undistorted"
[image2a]: ./test_images/straight_lines1.jpg "Road Transformed"
[image2b]: ./output_images/undistorted.jpg "Road UndisTransformed"
[image3a]: ./output_images/gray_binary.jpg "Binary Example"
[image3b]: ./output_images/color_binary.jpg "Binary Example2"
[image4]: ./output_images/srcwarped.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[videogif]: https://raw.githubusercontent.com/charlielito/CarND-Advanced-Lane-Lines/master/project_video_result.gif "S"

---

### Camera Calibration

#### Computation of camera matrix and distortion coefficients

The code for this step is contained in the script located in `./code/calibration.py`. It has a function called `calibrate_camera` that receives a list of images to base the calibration on and the `nx` and `ny` of the chessboard and finds the calibration parameters.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. There is a flag in the function  `show_images` to show each image and the chessboard detection one by one for debugging.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function if and only if there are more than 5 pattern detected in the image dataset. This because with less images the calibration is not that good. At the end the function returns as a tuple the camera matrix and the distortion coefficients, or a tuple of `None` if the last check was not succesful.

I used that function to get the parameters and if found, save them as a `.npz` in the `code` folder. I applied this distortion correction to the test image using the `cv2.undistort()` function after reading the `calibration.npz` file with the `get_camera_calibration` function and obtained this result (runing the script `code/calibration_test.py`): 

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

I used a combination of color and gradient thresholds to generate a binary image: function `get_binary_image` in the script mentioned at the beginning of this section. I used the HLS space for color thresholding since in this space the yellow lines and white lines can be detected using the S channel without much problem. I also used the Sobel transform in the `x` direction since this direction is good for detecting vertical lines. This function receives the tresholding limits as a tuple, for S channel and `x` Sobel absolute values tresholding. The values used are `(130,250)` and `(20,120)` respectively as you can see in line 215 of `base_line.py`.

Here's an example of my output for this step

![alt text][image3a]

And here you can see the contribution of each tresholding technique to the previous image: blue are the pixels from S treshold and green from Sobel treshold.
![alt text][image3b]


#### 3. Perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 320, 720      | 
| 599, 446      | 320, 0        |
| 681, 446      | 960, 0        |
| 1122, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Modifications to work with video


#### 2. Final video output.

Here's a [link to my video result](./output.mp4)

![alt text][videogif]

---

### Discussion

#### 1. Problems / issues in the implementation of this project. Failures? More robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
