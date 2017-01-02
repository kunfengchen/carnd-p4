import argparse
import cv2

# import p4 files
from camera_cal import calibrate_camera, undistort, show_undistorted_images
from color_pipe import color_pipeline, show_threshold_images
from warp import warp_image, cal_warp_points
from line import get_line_histogram, show_historgram

"""
Create a image process pipe line for line detection
"""

def calibrate():
    print("Calirating the camera ...")
    calibrate_camera(visual=False)


def detect_line_image(img_name, visual=False):
    print("Detecting the lines ...")
    # load the image
    img = cv2.imread(img_name)
    # undistort the image
    undist_img = undistort(img, visual=False)
    # if visual:
    #    show_undistorted_images(img, undist_img)
    # apply thresholding for sobelx and HLS color space
    thresh_img = color_pipeline(img)
    # if visual:
    #    show_threshold_images(img, thresh_img)
    # apply warp
    src_ps, dst_ps = cal_warp_points(img)
    warped_img, M, Minv = warp_image(thresh_img, src_ps, dst_ps)
    if visual:
        show_threshold_images(img, warped_img)
    # Line histogram
    hist = get_line_histogram(warped_img)
    if visual:
        show_historgram(hist)


    #warp.show_two_images(img, warped,
    #                src_ps, dst_ps,
    #                title_src="Unditorted Image",
    #                title_dst="Warped Image")


def detect_line_video(video_name):
    pass


## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--calibrate_camera',
        help='Calibrate the camera')
    parser.add_argument(
        '--image',
        default="test_images/test1.jpg",
        help='image to be processed')
    parser.add_argument(
        '--video',
    default="project_video.mp4"
    )
    args = parser.parse_args()

    if args.calibrate_camera:
        calibrate()
    if args.image:
        detect_line_image(args.image, visual=True)
    if args.video:
        detect_line_video(args.video)