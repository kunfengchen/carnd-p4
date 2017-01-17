# Carnd-P4: Advanced Lane Finding
A software pipeline to identify the lane boundaries in video from a front-facing camera on a car.
The camera calibration images, test road images, and videos are available in this [repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

### Camera Calibration
OpenCV functions are implemented in in [camera_cal.py](camera_cal.py) to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository.

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images (in the camera_cal folder in the repository). The function is implemented in calibrate_camera() that takes in numbers of corners in x and y directions and save the calibration matrix result in file camera_cal/camera_dist.p. The funtion uses provided images in camera_cal/calibraion*.jpg to caculate the results. The results are saved to file for pipeline process.

    
    ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)
    
    # Do camera calibration given object points and image points
    img_size = (img_shape[1], img_shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open(calibration_file , "wb" ) )
    
    
    python3 camera_cal.py --calibrate --visual
    Found corners on camera_cal/calibration10.jpg
    Found corners on camera_cal/calibration11.jpg
    Found corners on camera_cal/calibration7.jpg
    Found corners on camera_cal/calibration8.jpg
    Found corners on camera_cal/calibration16.jpg
    Found corners on camera_cal/calibration20.jpg
    Found corners on camera_cal/calibration13.jpg
    Found corners on camera_cal/calibration14.jpg
    Found corners on camera_cal/calibration3.jpg
    Found corners on camera_cal/calibration17.jpg
    Found corners on camera_cal/calibration2.jpg
    Found corners on camera_cal/calibration15.jpg
    Found corners on camera_cal/calibration6.jpg
    Found corners on camera_cal/calibration19.jpg
    Found corners on camera_cal/calibration12.jpg
    Found corners on camera_cal/calibration9.jpg
    Found corners on camera_cal/calibration18.jpg
   


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
The funciton is impletmented in [camera_cal.py](camera_cal.py). undistort() function uses the saved calibration matrix to undisotrt the image.

   ```
    if  dist_matrix is None:
        dist_matrix = load_dist_matrix()
    mtx = dist_matrix["mtx"]
    dist = dist_matrix["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)
   ```
   
Here are examples for undistorting an image.

python3 camera_cal.py --undistort camera_cal/calibration1.jpg:
![python3 camera_cal.py --undistort camera_cal/calibration1.jpg](examples/undistorted_image.png)

python3 camera_cal.py --undistort test_images/test4.jpg:
![python3 camera_cal.py --undistort test_images/test4.jpg](examples/undistorted_image_1.png)

#### Use Color Transforms
The function is impletmented in [color_pipe.py](color_pipe.py). color_pipeline takes in an input image and covert it to HLS color space. The lightness chanell is extrated and applied the Sobel in X direction with threshold between (20, 100), and scaled to between (0. 255), and becomes mask1. Extrat the saturation channel from the color space, applied threshold between (170, 255), and became mask2. Combine mask1 and mask2 to form the color_piplne process.

Here is an example for color pipeline process.

python3 color_pipe.py --colored test_images/test4.jpg:
![python3 color_pipe.py --colored test_images/test4.jpg](examples/colored_image.png)
Note: for debugging purpose, the green color shows mask1, and blue mask2. As you can see, combined mask1 and mask2 shows a longer left lane line.

#### Apply a Perspective Transform (Birds-eye View, Warp image)
The function is implemented in [warp.py](warp.py). warp_image() takes in an inputer image, source points, and destination points and warp the image into the perspective "birds-eye view." The function will return the inversed transform matrix for un-warping the image.

Here is an example for perspective transform.

python3 warp.py --warp test_images/test4.jpg:
![python3 warp.py --warp test_images/test4.jpg](examples/warped_image.png)
Here we want to make sure lef and right land lines are parallel by adusting the source points (in red dots).

#### Detect Lane Pixels and Fit to Find Lane Boundary
The function is implemented in [pipeline.py](pipeline.py). pipleline.py is the main file that apply all the sub-pipelines. After applying calibration, thresholding, and a perspective transform, we should have a binary image where teh lane lines stand out clearly. Next step is to find out exactly which pixels belong to the left line and which belong to the right line. The technique used here is to find peaks in a histogram.

First take a histogram alon all the columns in teh lower part of the image:
```
histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
```
Then find the two peaks representing the left and right lines x locations. Build find boxs around the locations to find non-zero pixels, and use the pixels to fit a second order polynomial for left and right lines, repsectively. The Line class in line.py is uesd for line information.

detect_line_image() takes in an input image and detect the lane lines. There are two way of detecting lane lines:
 * detect_lines_sliding_window() detects lane lines from the scratch using histogram.
 * detect_lines_previous detects lane lines with information infered from the previous detection.

The basic logic is to use detect_lines_sliding_window() at the beginning frames, which will take longer time, and then use that information to get the region of interest to use detect_lines_previous() for detecting line to save time. If detect_lines_previous() failed to obtain a good line, detect_lines_sliding_window() is used again from the scratch.

There are three methods used to determind if lines detected are good. The methods are implemented line.py inside Line class. check_line parallel() checks if two lines are parallel. check_similar_curvatures checks if two lines have similar curvatures, and check_distance checks if two lines have the distance.

#### Determine Curvature of The Lane and Vehicle Position with Respect to Center
Line class get_curvatgure_radius() calculates the lane curvature from the fitted line. You can find [a curvature tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). get_off_center_as_left() calculates the center position of the car. The function asumes the center x location is at pixcel number 805, and lane width is 950-pixel wide. The left line x position (at y in image bottom) is used to calculate the car center location. The unit is coverted to meter using the ratial of 3.7/700 meters per pixel.

#### Warp the Detected Lane Boundaries Back Onto the Original Image
It's time to project the detected lines back down onto the image. First draw the detected lane lines onto a warped blank image. Then warp the blank image back to original image using cv2.warpPerspective() with the inversed perspective matrix returned from the function warp_image() in warp.py. Finally, combine teh result with the original image.

That's it. That's the whole pipeline for detecting land lines using advanced techniques!

#### Whole Pipeline Diagram Views
The view is implemented in view.py. The two types of views: One is sliding-window view, and the other is previous-line view.
 * slidig-window views
  
  python3 pipeline.py --image test_images/test1.jpg --visual
  
  ![img](examples/dia_sliding_test1.png)
  
  python3 pipeline.py --image test_images/test1.jpg --visual
  
  ![img](examples/dia_sliding_test5.png)
 
 * previous-line view (video mode)
 
  python3 pipeline.py --visual
  
  ![img](examples/dia_previous.png)

### Pipeline (video)
After applying pipline in static images, now it's time to apply the pipline to video.

Here is the result:
[project_video_result.mp4](project_video_result.mp4)

### Final Thoughts
#### It's great to learn new techiques from OpenCV.
Learned somethnig new each day, especailly more familiar with Python Class, Numpy, OpenCV, and Matplotlib. Those are very useful tools to communicate the program and idea to people in visualzation. Finally got time to learn how subplot and GridSpec work.

#### How to Success
We need to be very carfull and fully tested for each pipeple. There are too much dependencies on one another. When we change one parameter, we better test it on all the pipelines.

Thanks to fellow classmate John Chen, he gave the idea of using diagnositc view for debugging the pipeline. We can ping piont the problem right away through the view will all the intermediate images displayed.

One more thing is if we can do a better job in a pipeline, we will save more time on the pileline that followed. The more is followed the more time is save. The corner cases will be drastically reduces.

#### Could Be Improved
There are a lot of fine tuning, assumptions, and hard coded pararmeters. It would be great to reduce those to for the pipeline to be more robust and apply to more situations.

For example, the perspective transformation source pionts is eyeballed and hard coded. This will take more time and manual effort to apply to a diffent view point of the camera. The center position is assumed after processing the video and eyeballed the results. The information is not very accuration and only for fun.

When changing the perspective tranformation source points, the fllowing piplelines are affected, too, since the resluting pixels are different. This requires more testing and make the funcitons less portable for other videos.

#### Hypothetical cases for Pipeline to Fail
For the uphill or downhill roads the pipleline may fail because the camera view to the road is different on the road vanishing point, which will affect the warp function source pionts that produce undesired results.

Different wether conditions may also cause the pipeline to fail because more (or less) sun light could could affect color appearence in the camera. This will affect the color pipeline to add more noise or fail to filter out the lines (might due to the threhold setting).

#### Overall
For my pipeline there is still a very long way to go to be in production. Too many hard-code pararmeter and threshold to be manually tuned to work correctly. This will be difficult to deploy since each car has different camera setup. Hope we can combine machine learing to solve the problem together with less fine tune process.
