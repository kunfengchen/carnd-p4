import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, figaspect

n_w = 4.0
n_h = 4.0
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 3/n_w, 3/n_h], xticks=[], yticks=[])
ax2 = fig.add_axes([3/n_w, 3/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax3 = fig.add_axes([3/n_w, 2/n_h, 1/n_w, 1/n_h], xticks=[])
ax4 = fig.add_axes([3/n_w, 1/n_h, 1/n_w, 1/n_h], xticks=[], yticks=[])
ax4.set_xlim(0, 1280)
ax4.set_ylim(720, 0)
ax5 = fig.add_axes([3/n_w, 0/n_h, 1/n_w, 1/n_h])


def show_images(img1, img2, img3, img4):
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img3, cmap='gray')
    ax4.imshow(img4, cmap='gray')


def show_plots(plots):
    for i, p in enumerate(plots):
        # plt.plot(p + i * 10)
        ax5.plot(p)

def show_pixels(xs, ys):
    ax4.plot(xs, ys, 'gx')

def show_xy(xs, ys):
    ax5.plot(xs, ys, 'go')

def show_fit_line(line, yvals):
    ax4.plot(line.best_fit_p(yvals), yvals, linewidth=3)

def show():
    plt.show()


if __name__ == "__main__":
    img1 = np.zeros((50, 50, 3))
    img1[:, :, 0] = 1
    img2 = np.zeros((25, 25, 3))
    img2[:, :, 1] = 1
    img3 = np.zeros((25, 25, 3))
    img3[:, :, 2] = 1
    show_images(img1, img2, img3)
