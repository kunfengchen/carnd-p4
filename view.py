import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from warp import cal_warp_points
from io import BytesIO


class View:

    def __init__(self):
        n_w = 4.0
        n_h = 4.0
        self.fig = plt.figure()
        self.ax1_m = self.fig.add_axes([0, 1/n_h, 3/n_w, 3/n_h])

        self.ax1_v = self.fig.add_axes([3/n_w, 3/n_h, 1/n_w, 1/n_h])
        self.ax2_v = self.fig.add_axes([3/n_w, 2/n_h, 1/n_w, 1/n_h])
        self.ax3_v = self.fig.add_axes([3/n_w, 1/n_h, 1/n_w, 1/n_h])
        self.ax3_v.set_xlim(0, 1280)
        self.ax3_v.set_ylim(720, 0)
        self.ax4_v = self.fig.add_axes([3/n_w, 0/n_h, 1/n_w, 1/n_h])

        self.ax1_h = self.fig.add_axes([0/n_w, 0/n_h, 1/n_w, 1/n_h])
        self.ax2_h = self.fig.add_axes([1/n_w, 0/n_h, 1/n_w, 1/n_h])
        self.ax3_h = self.fig.add_axes([2/n_w, 0/n_h, 1/n_w, 1/n_h])
        self.fig.subplots_adjust(wspace=2, hspace=2)
        self.init_axes()


    def init_axes(self):
        for a in self.fig.get_axes():
            a.clear()
            a.set_xticks([])
            a.set_yticks([])


    def show_main_images(self, img1):
        self.ax1_m.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))


    def show_vertical_images(self, img1, img2, img3, img4):
        if img1 is not None:
            self.ax1_v.imshow(img1, cmap='gray')
        src_ps, dst_ps = cal_warp_points(img2)
        for p in src_ps:
            self.ax1_v.plot(p[0], p[1], 'ro')
        if img2 is not None:
            self.ax2_v.imshow(img2, cmap='gray')
        if img3 is not None:
            self.ax3_v.imshow(img3, cmap='gray')
        if img4 is not None:
            self.ax4_v.imshow(img4, cmap='gray')


    def show_horizontcal_images(self, img1, img2, img3):
        if img1 is not None:
            self.ax1_h.imshow(img1)
        if img2 is not None:
            self.ax2_h.imshow(img2)
        if img3 is not None:
            self.ax3_h.imshow(img3)


    def show_plots(self, plots):
        for i, p in enumerate(plots):
            plt.plot(p + i * 10)
            self.ax4_v.plot(p)


    def show_pixels(self, xs, ys):
        self.ax3_v.plot(xs, ys, 'gx')


    def show_xy(self, xs, ys):
        self.ax4_v.plot(xs, ys, 'go')


    def show_fit_line(self, line):
        if line.current_fit_p is not None:
            self.ax2_v.plot(line.current_fit_p(line.yvals), line.yvals, linewidth=3)


    def show_found_boxs(self, found_boxs):
        for b in found_boxs[0]:
            self.ax2_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='yellow',
                          fill=None,
                          linewidth=2))
        for b in found_boxs[1]:
            self.ax2_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='pink',
                          fill=None,
                          linewidth=2))
        for b in found_boxs[2]:
            self.ax2_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='red',
                          fill=None,
                          linewidth=2))


    def show_roi_img(self, img, l):
        """
        Show the region of interest image
        :param img: the input image
        :param l: l=0 for left lane and l=1 for right lane
        :return:
        """
        self.ax3_v.imshow(img)


    def show(self):
        # plt.ion()
        plt.show()


    def draw(self):
        plt.draw()


    # REF: https://forum.omz-software.com/topic/1961/putting-a-matplotlib-plot-image-into-a-ui-imageview
    def get_image(self):
        """
        Get the polt in image
        :return: The image fo the plot
        """
        b = BytesIO()
        plt.savefig(b)
        nbarray = np.fromstring(b.getvalue(), np.uint8)
        return cv2.imdecode(nbarray, cv2.IMREAD_COLOR)


    def clear(self):
        """
        Clear all the axes in the figure
        :return:
        """
        self.init_axes()


    def close(self):
        """
        Close the figure
        :return:
        """
        plt.close(self.fig)


if __name__ == "__main__":
    img1 = np.zeros((50, 50, 3))
    img1[:, :, 0] = 255
    img2 = np.zeros((25, 25, 3))
    img2[:, :, 1] = 255
    img3 = np.zeros((25, 25, 3))
    img3[:, :, 2] = 255
    img4 = np.zeros((25, 25, 3))
    img4[:,:,:] = (134, 230, 11)

    view = View()
    view.show_main_images(img1.astype(np.uint8))
    view.show_horizontcal_images(img4.astype(np.uint8), img2, img3)
    view.show_vertical_images(img1.astype(np.uint8), img2, img3, img3)
    #view.draw()

    view = View()
    view.show_main_images(img2.astype(np.uint8))
    view.show_horizontcal_images(img1.astype(np.uint8), img2, img3)
    view.show_vertical_images(img1.astype(np.uint8), img2, img3, img3)
    view.show()

