import argparse
import cv2
import numpy as np
import scipy.signal as signal

# import p4 files
from camera_cal import calibrate_camera, undistort, show_undistorted_images
from color_pipe import color_pipeline, show_threshold_images
from warp import warp_image, cal_warp_points
from line import Line, get_line_histogram, show_historgram
from view import show_main_images, show_horizontcal_images,\
    show_vertical_images, show_plots, show_pixels,\
    show_fit_line, show_find_boxs,\
    show_find_boxs, show_xy, show

"""
Create a image process pipe line for line detection
"""

def draw_line_image(warped_img, lines):
    # Create an image to draw the lines on

    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(
        [np.transpose(np.vstack([lines[0].best_fit_p(lines[0].yvals), lines[0].yvals]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([lines[1].best_fit_p(lines[1].yvals), lines[1].yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp



def calibrate():
    print("Calirating the camera ...")
    calibrate_camera(visual=False)


def detect_line_image(img_name, visual=False):
    print("Detecting the lines ...")
    # load the image
    input_img = cv2.imread(img_name)
    # undistort the image
    undist_img = undistort(input_img, visual=False)
    # if visual:
    #    show_undistorted_images(img, undist_img)
    # apply thresholding for sobelx and HLS color space
    thresh_img = color_pipeline(input_img)
    # if visual:
    #    show_threshold_images(img, thresh_img)
    # apply warp
    src_ps, dst_ps = cal_warp_points(input_img)
    warped_img, M, Minv = warp_image(thresh_img, src_ps, dst_ps)
    if visual:
        show_vertical_images(input_img, thresh_img, warped_img, warped_img)

    ### Sliding window to detect lands
    # Line histogram
    n_frame = 5 # number of frames
    lines = [Line(), Line()] # left and right land
    find_boxs = []  # keep track of boxs for finding pixels
    peak_offset = 50
    land_detect_width = 100

    for frame_n in range(int(n_frame)):
        frame_h = int(warped_img.shape[0]/n_frame)  # frame height

        frame_y_1 = frame_h*(n_frame-frame_n-1)
        frame_y_2 = frame_y_1 + frame_h

        # print (frame_n, frame_y_1, frame_y_2)
        frame_img = warped_img[
                      frame_y_1:
                      frame_y_2,:]
        hist = get_line_histogram(frame_img)

        n_win = 100 # number of windows
        ### r = np.exp(-t[:1000]/0.05) gaussian
        # smooth the histgram
        norm_hist = np.convolve(hist, np.ones(n_win)/n_win, mode='same')
        norm_peak_ind = signal.find_peaks_cwt(norm_hist, np.arange(90, 100))
        n_peaks = len(norm_peak_ind)
        peak_ind = 0  # peak index
        print("Found " + str(n_peaks) + "peaks at window " + str(frame_n))
        if n_peaks > 1:
            # clip at 20
            # norm_hist[norm_hist < 20] = 0

            # Find pixels

            for l in range(0, 2): # left and right lanes
                peakx = norm_peak_ind[peak_ind] - peak_offset;
                peak_ind += 1
                if peakx < 0 & peak_ind < n_peaks: # left boundary of x
                    # get the next peak
                    peakx = norm_peak_ind[peak_ind] - peak_offset;
                    peak_ind += 1

                find_box = (peakx, frame_y_1, land_detect_width, frame_h)
                find_boxs.append(find_box)
                print(find_box)
                non_zero_pixels = cv2.findNonZero(
                    np.array(frame_img[:, find_box[0]:find_box[0]+find_box[2]], np.uint8))

                if non_zero_pixels is not None:
                    found_xs = non_zero_pixels[:, 0, 0]
                    found_xs = found_xs + peakx  # add the shift done by slicing
                    found_ys = non_zero_pixels[:, 0, 1]
                    found_ys = found_ys + frame_y_1  # add the frame height back
                    lines[l].allx.extend(found_xs)
                    lines[l].ally.extend(found_ys)


    # Fit lines
    for l in range(0 ,2): # left and right lands
        lines[l].fit_poly_line()

    color_warp_img = draw_line_image(warped_img, lines)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp_img, Minv, (color_warp_img.shape[1], color_warp_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    if visual:
        #show_xy(hist_peak_ind, hist[hist_peak_ind])
        show_xy(norm_peak_ind, norm_hist[norm_peak_ind])
        # show_xy(lines[0].allx, frame_y_2 - lines[0].ally - frame_h*(n_frame-frame_n-1))
        for l in range(0 ,2): # left and right lands
            show_pixels(np.array(lines[l].allx), np.array(lines[l].ally))
            show_fit_line(lines[l])
        # print(find_boxs)
        show_find_boxs(find_boxs)
        show_plots((hist, norm_hist))
        show_main_images(result)
        show_horizontcal_images(color_warp_img, newwarp, newwarp)
        show()

    return result


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
        default="test_images/test2.jpg",
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