import numpy as np
import matplotlib.pyplot as plt

### REF: ported from the class material

# Define a class to receive the characteristics of each line detection
class Line():

    # Y values to cover same y-range as image
    yvals = np.linspace(0, 100, num=101)*7.2
    #the find box that detected the line pixels
    # found_boxs[0] is left lines
    # found_boxs[1] is for right lines
    # found_boxs[2] is for bad lines
    found_boxs = None
    # meters per pixel in y dimension
    ym_per_pix = 30/720
    # meteres per pixel in x dimension
    xm_per_pix = 3.7/700

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # number of failed detection so far
        self.n_failed_detection = 0
        # threshold for failed detection
        self.failed_threshold = 15
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #ploynamail 1d
        self.best_fit_p = None
        #polynomial coefficients in meter for the most recent fit
        self.best_fit_meter = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #ploynamail 1d
        self.current_fit_p = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []


    def fit_poly_line(self):
        if len(self.allx) < 2 or len(self.ally) <2:
            self.detected = False
        else:
            last_fit = self.current_fit
            self.current_fit = np.polyfit(self.ally, self.allx, 2)
            self.current_fit_p = np.poly1d(self.current_fit)
            self.recent_xfitted = self.current_fit_p(Line.yvals)

            if self.best_fit is None:  # the first estimate
                self.best_fit = self.current_fit
                self.detected = True
            else:
                mse_thred = 200  # threshold in mse for bad fitting
                # self.diffs = last_fit - self.current_fit
                self.diffs = self.best_fit - self.current_fit
                mse = np.mean(self.diffs**2)
                # print("fit diffs", self.diffs.shape, self.diffs, mse)
                if (mse < mse_thred):
                    k = 0.9
                    #self.best_fit = \
                    #    (self.best_fit *(self.n_lines -1) + self.current_fit)/self.n_lines
                    self.best_fit = self.best_fit * k + self.current_fit * (1-k)
                    self.detected = True
                else:
                    # print(" bad fit")
                    # don't update the best_fit
                    self.add_one_fail_detection()
                    self.detected = False
            if self.detected:
                self.best_fit_p = np.poly1d(self.best_fit)
                self.fit_poly_line_meter()
        self.clear_detected_xs()
        self.clear_detected_ys()


    def add_detected_xs(self, xs):
        self.allx.extend(xs)


    def add_detected_ys(self, ys):
        self.ally.extend(ys)


    def clear_detected_xs(self):
        self.allx = []


    def clear_detected_ys(self):
        self.ally = []


    def fit_poly_line_meter(self):
        # Define conversions in x and y from pixels space to meters

        self.best_fit_meter = np.polyfit(
            Line.yvals*Line.ym_per_pix, self.best_fit_p(Line.yvals) * Line.xm_per_pix, 2)


    def get_curvature_radius(self, yval):
        curve_rad = 0
        if self.best_fit_meter is not None:
            curve_rad = ((1 + (2*self.best_fit_meter[0]*yval + self.best_fit_meter[1])**2)**1.5) \
                            /np.absolute(2*self.best_fit_meter[0])
        return curve_rad


    def get_off_center_as_left(self, yval=720, xcenter=805, lane_width=950):
        """
        Get how far away from the center lane as a left lane
        :return: the meter away from the center. puls number means to the left
        """
        left_x = self.best_fit_p(yval)
        left_center = left_x + lane_width/2
        off_center_meter = (xcenter-left_center) * self.xm_per_pix
        return off_center_meter


    def check_simlilar_curvatures(self, line, margin=0.15, yval=360):
        """
        Check that tow lines have similar curvature
        :param line: the other line to compare
        :param margin: margin threshold in percentage
        :param yval: the y value to check the curvature
        :return: True if the curvature difference is within the margin, False otherwise
        """
        cur1 = self.get_curvature_radius(yval)
        cur2 = line.get_curvature_radius(yval)
        dif = abs(cur1-cur2)
        dif_percent = dif/max(cur1, cur2)
        # print(" check curvatures:", cur1, cur2, dif, dif_percent)
        result = True
        if dif_percent > margin:
            result = False
        return result


    def check_distance(self, line, distance=800, margin=0.05, yval=720):
        """
        Check that tow lines are separated by approximately the right distance horizontally
        :param line: the other line to compare
        :param margin: margin threshold in percentage
        :param yval: the y value to check the distance
        :return: True is roughly the distance, False otherwise
        """
        dist = abs(line.best_fit_p(yval) - self.best_fit_p(yval))
        error = abs(dist - distance)
        error_percent = error/distance
        result = True
        if error_percent > margin:
            result = False
        return result


    def check_parallel(self, line, distance=850, margin=0.05):
        """
        Check that two lines are roughly parallel
        :param line: The other line to compare
        :param margin: margin threshold in percentage
        :return: True if roughly parallel, False otherwise
        """
        yvals = np.linspace(0, 100, num=11)*7.2
        result = True
        for y in yvals:
            if not self.check_distance(line):
                result = False
                break
        return result


    def get_best_roi(self):
        """
        Get the region of the interest using the area of best fit line
        :return:
        """
        yvals = np.linspace(0, 100, num=11)*7.2
        w = 20 # half width of the ROI
        roi = []
        for y in yvals:
            roi.append([self.best_fit_p(y)-w, y])
        for y in yvals[::-1]:
            roi.append([self.best_fit_p(y)+w, y])
        return roi


    def add_one_fail_detection(self):
        self.n_failed_detection += 1


    def should_start_new_line(self):
        """
        Check to see to the line still reusable for the next frame detection
        If not, discard this one and use a new instance.
        :return:
        """
        return self.n_failed_detection > self.failed_threshold


    # Class method
    def get_roi():
        """
        Get the region of the interest using find boxes
        :return: left and right line rois
        """
        roi = [[],[]]  #left and right lines

        for l in (0, 1):
            boxs = Line.found_boxs[l]
            for b in (0, -1): # first and last box
                point = [boxs[b][0],boxs[b][1]]
                roi[l].append(point)
            for b in (-1, 0):
                point = [boxs[b][0]+boxs[b][2], boxs[b][1]+boxs[b][3]]
                roi[l].append(point)
        return roi
