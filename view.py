import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from warp import cal_warp_points

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


### Ported from http://www.icare.univ-lille1.fr/node/1141
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf


### Ported from http://www.icare.univ-lille1.fr/node/1141
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )


class View:

    def __init__(self):
        n_w = 4.0
        n_h = 4.0
        self.fig = plt.figure()
        self.ax1_m = self.fig.add_axes([0, 1/n_h, 3/n_w, 3/n_h], xticks=[], yticks=[])

        self.ax1_v = self.fig.add_axes([3/n_w, 3/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
        self.ax2_v = self.fig.add_axes([3/n_w, 2/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
        self.ax3_v = self.fig.add_axes([3/n_w, 1/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
        self.ax3_v.set_xlim(0, 1280)
        self.ax3_v.set_ylim(720, 0)
        self.ax4_v = self.fig.add_axes([3/n_w, 0/n_h, 1/n_w, 1/n_h])

        self.ax1_h = self.fig.add_axes([0/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
        self.ax2_h = self.fig.add_axes([1/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
        self.ax3_h = self.fig.add_axes([2/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])

    def show_main_images(self, img1):
        self.ax1_m.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    def show_vertical_images(self, img1, img2, img3, img4):
        self.ax1_v.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        src_ps, dst_ps = cal_warp_points(img2)
        for p in src_ps:
            self.ax2_v.plot(p[0], p[1], 'ro')
        self.ax2_v.imshow(img2, cmap='gray')
        self.ax3_v.imshow(img3, cmap='gray')
        self.ax4_v.imshow(img4, cmap='gray')

    def show_horizontcal_images(self, img1, img2, img3):
        self.ax1_h.imshow(img1)
        self.ax2_h.imshow(img2)
        self.ax3_h.imshow(img3)


    def show_plots(self, plots):
        for i, p in enumerate(plots):
            # plt.plot(p + i * 10)
            self.ax4_v.plot(p)

    def show_pixels(self, xs, ys):
        self.ax3_v.plot(xs, ys, 'gx')

    def show_xy(self, xs, ys):
        self.ax4_v.plot(xs, ys, 'go')

    def show_fit_line(self, line):
        self.ax3_v.plot(line.current_fit_p(line.yvals), line.yvals, linewidth=3)

    def show_found_boxs(self, found_boxs):
        for b in found_boxs[0]:
            self.ax3_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='yellow',
                          fill=None,
                          linewidth=2))
        for b in found_boxs[1]:
            self.ax3_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='pink',
                          fill=None,
                          linewidth=2))
        for b in found_boxs[2]:
            self.ax3_v.add_patch(
                Rectangle((b[0], b[1]), b[2], b[3],
                          color='red',
                          fill=None,
                          linewidth=2))

    def show(self):
        # plt.ion()
        plt.show()


    def draw(self):
        plt.draw()


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

