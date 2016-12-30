import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import camera_cal

def warp_img(img, src, dst):
    """
    Perspective transform a image
    :param img: Input image
    :param src: src points
    :param dst: dst points
    :return: warped image, M
    """
    img_size = (img.shape[1], img.shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M

def show_two_images(src, dst,
                    src_points, dst_points,
                    title_src="Original Image",
                    title_dst="Second Image",
                    ):
    """
    Show orginal and warped images
    :param src: The original image
    :param dst: The warp image
    :return:
    """
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(src_rgb)
    for p in src_points:
        ax1.plot(p[0], p[1], 'ro')
    ax1.set_title(title_src, fontsize=30)
    ax2.imshow(dst_rgb)
    for p in dst_points:
        ax2.plot(p[0], p[1], 'ro')
    ax2.set_title(title_dst, fontsize=30)
    plt.show()

## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--warp',
        default="test_images/test1.jpg",
        help='Warp the image')
    args = parser.parse_args()
    src_ps = np.float32([[717, 435], # top right
                      [1125, 705], # bottom right
                      [258, 705],  # bottom left
                      [615, 435]])  # top left
    dst_ps = np.float32([[1125, 200], # top right
                      [1125, 700], # bottom right
                      [258, 700],  # bottom left
                      [258, 200]])  # top left
    img = cv2.imread(args.warp)
    undistorted = camera_cal.undistort(img)
    warped, M = warp_img(undistorted, src_ps, dst_ps)
    show_two_images(img, warped,
                    src_ps, dst_ps,
                    title_src="Unditorted Image",
                    title_dst="Warped Image")