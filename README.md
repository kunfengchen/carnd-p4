# Carnd-P4: Advanced Lane Finding (In Progress ...)
A software pipeline to identify the lane boundaries in video from a front-facing camera on a car.
The camera calibration images, test road images, and videos are available in this [repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

### Camera Calibration
OpenCV functions or other methods are used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository.

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images (in the camera_cal folder in the repository).

### Pipeline (single images)
for a series of test images (in the test_images folder in the repository):

* Apply the distortion correction to the raw image.
* Use color transforms, gradients, etc., to create a thresholded binary image.camer
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

#### Apply Distortion
The funciton is impletmented in [camera_cal.py](camera_cal.py). calibrate_camera() takes in numbers of corners in x and y directions and save the calibration matrix result in file camera_cal/camera_dist.p. The funtion uses provided images in camera_cal/calibraion*.jpg to caculate the results. undistort() function uses the saved calibration matrix to undisotrt the image.

Here are examples for undistorting an image.
python3 camera_cal.py --undistort camera_cal/calibration1.jpg:
![python3 camera_cal.py --undistort camera_cal/calibration1.jpg](examples/undistorted_image.png)
python3 camera_cal.py --undistort test_images/test4.jpg:
![python3 camera_cal.py --undistort test_images/test4.jpg](examples/undistorted_image_1.png)

#### Use Color Transforms
The function is impletmented in [color_pipe.py](color_pipe.py). color_pipeline takes in the input image and covert it to HLS color space. The lightness chanell is extrated and applied the Sobel in X direction with threshold between (20, 100), and scaled to between (0. 255), and becomes mask1. Extrat the saturation channel from the color space, applied threshold between (170, 255), and became mask2. Combine mask1 and mask2 to form the color_piplne process.

Here is an example for color pipeline process.
python3 color_pipe.py --colored test_images/test4.jpg:
![python3 color_pipe.py --colored test_images/test4.jpg](examples/colored_image.png)
(Note: for debugging purpose, the green color shows mask1, and blue mask2)





### Pipeline (video)
