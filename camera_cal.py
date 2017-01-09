import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

### REF: https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
### REF: ported from class material

# Default calibration file name
CALIBRATION_FILE_NAME = "camera_cal/camera_dist.p"
# Default number of corners in x
CORNERS_X = 9
# Default number of corners in y
CORNERS_Y = 6
# Default calibration file list for glob
CALIBRATION_FILE_LIST = "camera_cal/calibration*.jpg"

def calibrate_camera(corners_x = CORNERS_X,
                     corners_y = CORNERS_Y,
                     calibration_file = CALIBRATION_FILE_NAME,
                     visual=False):
    """
    :param corners_x: The number of conners in x direction
    :param corners_y: The number of corners in y direction
    :param calibration_file: The file name to save the calibration
    :param visual: Show the calibration visualization
    :return:
    """

    ## extract object points and image points for camera calibration
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corners_y*corners_x,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(CALIBRATION_FILE_LIST)
    test_images = [] # images for testing undistort
    img_shape = None

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_shape = img.shape
        if visual:
            test_images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)

        # If found, add object points, image points
        if ret == True:
            print("Found corners on {}".format(fname))
            objpoints.append(objp)
            imgpoints.append(corners)

            if visual:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corners_x, corners_y), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)
    if visual:
        cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    img_size = (img_shape[1], img_shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open(calibration_file , "wb" ) )

    if visual:
        # img = cv2.imread('camera_cal/calibration1.jpg')
        for img in test_images[0:9]:
            img_size = (img.shape[1], img.shape[0])

            dst = cv2.undistort(img, mtx, dist, None, mtx)

            # Visualize undistortion
            show_undistorted_images(img, dst)


def undistort(img,
        dist_matrix = None,
        visual=False):
    """
    Undistort a image using calibratoin saved in a specified file
    :param calibration_file: The file with calibration data
    :param img: The image to be undistorted
    :param visual: If true, show the progress
    :return: The distorted image
    """
    if  dist_matrix is None:
        dist_matrix = load_dist_matrix()
    mtx = dist_matrix["mtx"]
    dist = dist_matrix["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    if visual:
        show_undistorted_images(img, dst)
    return dst

def load_dist_matrix(
        calibration_file = CALIBRATION_FILE_NAME):
    return pickle.load(open(calibration_file, mode='rb'))

def show_undistorted_images(org, undist):
    """
    Show orginal and undistorted images
    :param org: The original image
    :param undist: The undistorted image
    :return:
    """
    orgr = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
    undistr = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(orgr)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistr)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--calibrate', default=False,
                        action='store_true',
                        help='Calibrate the camera')
    parser.add_argument('--undistort', default="camera_cal/calibration1.jpg",
                        help='Undistort the image')
    parser.add_argument('--visual', default=False,
                        action='store_true',
                        help='Show the process visualization')
    args = parser.parse_args()

    if args.calibrate:
        calibrate_camera(visual=args.visual)

    if args.undistort:
        undistort(cv2.imread(args.undistort), visual=True)


if __name__ == '__main__':
    main()
