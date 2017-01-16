# Carnd-P4: Advanced Lane Finding (In Progress ...)
A software pipeline to identify the lane boundaries in video from a front-facing camera on a car.
The camera calibration images, test road images, and videos are available in this [repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

### Camera Calibration
OpenCV functions or other methods are used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository.

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images (in the camera_cal folder in the repository).

### Pipeline (single images)
for a series of test images (in the test_images folder in the repository):

* Apply the distortion correction to the raw image.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

#### Apply Distortion
The funciton is impletmented in [camera_cal.py](camera_cal.py)

### Pipeline (video)
