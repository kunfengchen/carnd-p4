import argparse
import cv2
import numpy as np
import scipy.signal as signal

# import p4 files
from camera_cal import calibrate_camera, undistort, load_dist_matrix, show_undistorted_images
from color_pipe import color_pipeline, show_threshold_images
from warp import warp_image, cal_warp_points
from line import Line, get_line_histogram, show_historgram
from view import show_main_images, show_horizontcal_images,\
    show_vertical_images, show_plots, show_pixels,\
    show_fit_line, show_found_boxs, show_xy, show

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


def check_find_box(lr, x, found_boxs):
    """
    Check if x is a good left lane find box based on previous one
    :param lr: lr = 0 left, lr=1 right lane line
    :param x: the top-left cornor of new find box
    :param found_boxs: the previous found boxs
    :return: True if x is a good next find box, False otherwise
    """
    # threshold ragne for stacking up the find boxs
    range = 50
    # maximum x for right lane
    left_max_x = 600
    # minimum x disstance between left and right lanes
    lane_dist_min = 800
    left_boxs = found_boxs[0]  # left boxs
    right_boxs = found_boxs[1]  # right boxs
    left_box = None if len(left_boxs) == 0 else left_boxs[-1] # previous right box
    right_box = None if len(right_boxs) == 0 else right_boxs[-1] # previous right box

    # print("check box:", lr, x, found_boxs)
    if x < 0:  # outside boundary
        return False

    if lr == 0: # check for left line
        if (left_box is None):
            # print("The first right x")
            return True if x < left_max_x else False
        # Check if the box stacking up nicely
        left = left_box[0] - range
        right = left_box[0] + left_box[2] + range
        if x > left and x < right:
            # print("On the stack")
            return True
    else:  # check right box
        if left_box is not None:
            # For example: Frame # 70
            # print("right lane: check min dist" , x, left_box[0], lane_dist_min)
            if x > left_box[0] + lane_dist_min: # not too close to right lane
                return True

        # Check if stack up nicely
        if right_box is not None:
            left = right_box[0] - range
            right = right_box[0] + right_box[2] + range
            if x > left and x < right:
                # print("On the stack")
                return True

    return False


def detect_line_image_file(file_name, visual=False):
    # load the image from file
    input_img = cv2.imread(file_name)
    return detect_line_image(input_img, visual=visual)


def detect_line_image(
        input_img,
        dist_matrix=None,
        visual=False):
    # print("Detecting the lines ...")
    # undistort the image
    undist_img = undistort(input_img, dist_matrix=dist_matrix, visual=False)
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

    ### Sliding window to detect lanes
    # Line histogram
    n_frame = 10 # number of frames
    lines = [Line(), Line()] # left and right lane
    found_boxs = [[], [], []] # keep track of left, right, and bad boxs for finding pixels
    is_good_box = False
    peak_offset = 50
    lane_detect_width = 100

    for frame_n in range(int(n_frame*10/10)):  # skip top portion
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
        is_good_left_box = False  # If left box has been good.
        # If the find_box good or not
        # print("Found " + str(n_peaks) + " peaks at window " + str(frame_n))
        ### Find pixels
        if n_peaks > 0:
            for l in range(0, 2): # left and right lane lines
                if (l == 1): # right lane line
                    if peak_ind == n_peaks : # Didn't find a good box for left lane line
                        peak_ind = 0  # reset peak_ind from the beginnig for the right lane line
                # print("frame ", frame_n, "n_peaks", n_peaks, "l=" + str(l) + " peak_ind=" + str(peak_ind))
                while not is_good_box and peak_ind < n_peaks: # advance to the next good box
                    # get the next peak
                    peakx = norm_peak_ind[peak_ind] - peak_offset
                    peak_ind += 1
                    find_box = (peakx, frame_y_1, lane_detect_width, frame_h)
                    is_good_box = check_find_box(l, peakx, found_boxs)
                    if not is_good_box:
                        if l == 1:  # both lane lines failed
                            found_boxs[2].append(find_box)  # keep track of bad boxs, also


                # print("after while", l, is_good_box, peak_ind)
                if is_good_box:
                    found_boxs[l].append(find_box)
                    non_zero_pixels = cv2.findNonZero(
                        np.array(frame_img[:, find_box[0]:find_box[0]+find_box[2]], np.uint8))
                    if non_zero_pixels is not None: # add found pixels into the line
                        found_xs = non_zero_pixels[:, 0, 0]
                        found_xs = found_xs + peakx  # add the shift done by slicing
                        found_ys = non_zero_pixels[:, 0, 1]
                        found_ys = found_ys + frame_y_1  # add the frame height back
                        lines[l].allx.extend(found_xs)
                        lines[l].ally.extend(found_ys)
                is_good_box = False


    # Fit lines
    for l in range(0 ,2): # left and right lanes
        lines[l].fit_poly_line()

    color_warp_img = draw_line_image(warped_img, lines)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp_img, Minv, (color_warp_img.shape[1], color_warp_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    # Write lane information
    info1 = "Radius of Curvature = " + str(3.0) + " (m)"
    info2 = "Vehicle is " + str(2.0) + " (m) left of center"
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, info1, (20, 50), font, 1, (255, 255, 255), 3)
    cv2.putText(result, info2, (20, 100), font, 1, (255, 255, 255), 3)

    if visual:
        #show_xy(hist_peak_ind, hist[hist_peak_ind])
        show_xy(norm_peak_ind, norm_hist[norm_peak_ind])
        # show_xy(lines[0].allx, frame_y_2 - lines[0].ally - frame_h*(n_frame-frame_n-1))
        for l in range(0 ,2): # left and right lanes
            show_pixels(np.array(lines[l].allx), np.array(lines[l].ally))
            show_fit_line(lines[l])
        # print(find_boxs)
        show_found_boxs(found_boxs)
        show_plots((hist, norm_hist))
        show_main_images(result)
        show_horizontcal_images(color_warp_img, newwarp, newwarp)
        show()

    return result


def detect_line_video(video_name):
    dist_matrix = load_dist_matrix()

    video_out_name = "out.mp4"
    cap = cv2.VideoCapture(video_name)
    count = 0

    # Video out
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_out_name, fourcc, 20.0, (1280, 720))
    while cap.isOpened():
        ret, frame = cap.read()
        if (count%1) == 0:
            if frame is not None:
                out_frame = detect_line_image(frame, dist_matrix=dist_matrix, visual=True)
                out.write(out_frame)
                cv2.imshow('window-name', out_frame)
                print("frame " + str(count))
        #    cv2.imwrite("frame%d.jpg" % count, frame)
        count += 1
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--calibrate_camera',
        help='Calibrate the camera')
    parser.add_argument(
        '--image',
        #default="test_images/test6.jpg",
        #default="frame190.jpg",
        default="frame570.jpg",
        help='image to be processed')
    parser.add_argument(
        '--video',
    # default="project_video.mp4"
        default="challenge_video.mp4"
    )
    args = parser.parse_args()

    if args.calibrate_camera:
        calibrate()
    if args.image:
        detect_line_image_file(args.image, visual=True)
    if args.video:
        detect_line_video(args.video)