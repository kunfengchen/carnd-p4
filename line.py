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

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # number of previous lines
        self.n_lines = 0
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
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.current_fit_p = np.poly1d(self.current_fit)
        self.n_lines += 1
        # calculate the average for previous lines
        if self.n_lines == 1:
            self.best_fit = self.current_fit

        else:
            self.best_fit = \
                (self.best_fit *(self.n_lines -1) + self.current_fit)/self.n_lines
        self.best_fit_p = np.poly1d(self.best_fit)

    def fit_poly_line_meter(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        self.best_fit_meter = np.polyfit(
            Line.yvals*ym_per_pix, self.best_fit_p(Line.yvals) *xm_per_pix, 2)

    def get_curvature_radius(self, yval):
        curve_rad = ((1 + (2*self.best_fit_meter[0]*yval + self.best_fit_meter[1])**2)**1.5) \
                        /np.absolute(2*self.best_fit_meter[0])
        return curve_rad


    def get_roi(self):
        """
        Get the region of the interest
        :return: left and right line rois
        """
        roi = [[],[]]  #left and right lines
        for l in (0, 1):
            for b in Line.found_boxs[l]:
                roi[l].append((b[0], b[1]))  # append (x, y)



"""
    Checking that they have similar curvature
    Checking that they are separated by approximately the right distance horizontally
    Checking that they are roughly parallel
"""

"""



right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)

right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                 /np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 3380.7 m    3189.3 m



"""

def get_line_histogram(img):
    histogram = np.sum(img, axis=0)
    # print(histogram.shape)
    # print(histogram)
    # plt.plot(histogram)
    return histogram

def show_historgram(hist):
    plt.plot(hist)
    plt.show()

