import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img, vertices, view=None):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    pts = np.array(vertices, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    #print(pts.shape, pts)
    cv2.fillPoly(mask, [pts], ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    if view is not None:
        view.ax4_v.imshow(mask)
    return masked_image

def check_line_qualities(lines):
    """
    Check if the left and right lane lines still maintaining good qualities:
      1. Similar distance
      2. Parallel
      3. Similar curvatures
    :param lines: the left and right lane lines
    :return: True if good qualities, False otherwise
    """
    res1 = lines[0].check_simlilar_curvatures(lines[1])
    # print(" check curvature:", res1)
    res2 = lines[0].check_distance(lines[1])
    # print(" check distance:", res2)
    res3 = lines[0].check_parallel(lines[1])
    # print(" check parallel:", res3)
    return (res1 and res2 and res3)


def get_line_histogram(img):
    histogram = np.sum(img, axis=0)
    # print(histogram.shape)
    # print(histogram)
    # plt.plot(histogram)
    return histogram


def show_historgram(hist):
    plt.plot(hist)
    plt.show()
