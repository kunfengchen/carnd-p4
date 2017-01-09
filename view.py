import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from warp import cal_warp_points


n_w = 4.0
n_h = 4.0
fig = plt.figure()
ax1_m = fig.add_axes([0, 1/n_h, 3/n_w, 3/n_h], xticks=[], yticks=[])

ax1_v = fig.add_axes([3/n_w, 3/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax2_v = fig.add_axes([3/n_w, 2/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax3_v = fig.add_axes([3/n_w, 1/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax3_v.set_xlim(0, 1280)
ax3_v.set_ylim(720, 0)
ax4_v = fig.add_axes([3/n_w, 0/n_h, 1/n_w, 1/n_h])

ax1_h = fig.add_axes([0/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax2_h = fig.add_axes([1/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax3_h = fig.add_axes([2/n_w, 0/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])


def show_main_images(img1):
    ax1_m.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

def show_vertical_images(img1, img2, img3, img4):
    ax1_v.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    src_ps, dst_ps = cal_warp_points(img2)
    for p in src_ps:
        ax2_v.plot(p[0], p[1], 'ro')
    ax2_v.imshow(img2, cmap='gray')
    ax3_v.imshow(img3, cmap='gray')
    ax4_v.imshow(img4, cmap='gray')

def show_horizontcal_images(img1, img2, img3):
    ax1_h.imshow(img1)
    ax2_h.imshow(img2)
    ax3_h.imshow(img3)


def show_plots(plots):
    for i, p in enumerate(plots):
        # plt.plot(p + i * 10)
        ax4_v.plot(p)

def show_pixels(xs, ys):
    ax3_v.plot(xs, ys, 'gx')

def show_xy(xs, ys):
    ax4_v.plot(xs, ys, 'go')

def show_fit_line(line):
    ax3_v.plot(line.best_fit_p(line.yvals), line.yvals, linewidth=3)

def show_found_boxs(found_boxs):
    for b in found_boxs[0]:
        ax3_v.add_patch(
            Rectangle((b[0], b[1]), b[2], b[3],
                      color='yellow',
                      fill=None,
                      linewidth=2))
    for b in found_boxs[1]:
        ax3_v.add_patch(
            Rectangle((b[0], b[1]), b[2], b[3],
                      color='pink',
                      fill=None,
                      linewidth=2))
    for b in found_boxs[2]:
        ax3_v.add_patch(
            Rectangle((b[0], b[1]), b[2], b[3],
                      color='red',
                      fill=None,
                      linewidth=2))

def show():
    plt.show()


if __name__ == "__main__":
    img1 = np.zeros((50, 50, 3))
    img1[:, :, 0] = 1
    img2 = np.zeros((25, 25, 3))
    img2[:, :, 1] = 1
    img3 = np.zeros((25, 25, 3))
    img3[:, :, 2] = 1
    show_main_images(img1.astype(np.uint8))
    show_horizontcal_images(img1, img2, img3)
    show_vertical_images(img1, img2, img3, img3)
    show()
