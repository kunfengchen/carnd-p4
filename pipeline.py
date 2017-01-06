import argparse
import cv2
import numpy as np
import scipy.signal as signal

# import p4 files
from camera_cal import calibrate_camera, undistort, show_undistorted_images
from color_pipe import color_pipeline, show_threshold_images
from warp import warp_image, cal_warp_points
from line import Line, get_line_histogram, show_historgram
from view import show_images, show_plots, show_xy, show

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
        #show_threshold_images(img, warped_img)
        show_images(img, thresh_img, warped_img, warped_img)

    ### Sliding window to detect lands
    # Line histogram
    n_frame = 20 # number of frames

    for frame_n in range(n_frame):
        frame_h = warped_img.shape[0]/n_frame  # frame height

        frame_y_1 = frame_h*(n_frame-frame_n-1)
        frame_y_2 = frame_h*(n_frame-frame_n)

        print (frame_y_1, frame_y_2)
        frame_img = warped_img[
                      frame_y_1:
                      frame_y_2,:]
        hist = get_line_histogram(frame_img)

        n_win = 100 # number of windows
        ### r = np.exp(-t[:1000]/0.05) gaussian
        # smooth the histgram
        norm_hist = np.convolve(hist, np.ones(n_win)/n_win, mode='same')
        norm_peak_ind = signal.find_peaks_cwt(norm_hist, np.arange(90, 100))
        peakx = norm_peak_ind[0] - 50;
        # clip at 20
        norm_hist[norm_hist < 20] = 0

        lines = [Line(), Line()] # left and right land
        land_detect_width = 150

        # Find pixels
        for l in range(0, 2):
            non_zero_pixels = cv2.findNonZero(np.array(frame_img[:, peakx:peakx+land_detect_width], np.uint8))

            if non_zero_pixels is not None:
                lines[l].allx.append(
                    non_zero_pixels[:, 0, 0] + peakx)  # add the shift done by slicing

                lines[l].ally.append(non_zero_pixels[:, 0, 1])
                # u_non_ys = warped1_img.shape[0] - non_ys_left # from drawing

    if visual:
        #show_xy(hist_peak_ind, hist[hist_peak_ind])
        show_xy(norm_peak_ind, norm_hist[norm_peak_ind])
        # show_xy(lines[0].allx, frame_y_2 - lines[0].ally - frame_h*(n_frame-frame_n-1))
        show_xy(np.array(lines[0].allx), frame_h - np.array(lines[0].ally))
        show_xy(np.array(lines[1].allx), frame_h - np.array(lines[1].ally))
        show_plots((hist, norm_hist))
        show()

        # show_polts((hist, norm_hist, dif_hist))
        # show_historgram(hist)



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