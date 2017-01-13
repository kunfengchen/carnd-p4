import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import camera_cal

### REF: ported from class material

def warp_image(img, src, dst):
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
    # The inversed
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv

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

def cal_warp_points(img):
    """
    Get the src and dst points for the input of the warp image
    :param img: The input image to get the warp points
    :return: src and dst points
    """
    img_w = img.shape[1]
    img_h = img.shape[0]
    x_top_right_offset = -30  # fine tune for parallel lines
    x_bottom_right_offset = 100  # fine tune for parallel lines
    x_bottom_left_offset = -100  # fine turn for parallel lines
    bottom_high_offset = -50  # fine tune for parallel lines
    # top_dis = img_w * 15 / 100 # the x distance for src top points
    top_dis = img_w * 18 / 100 # the x distance for src top points
    bottom_dis = img_w * 96/100 # the x distacen for src bottom points
    # top_y = img_h * 62 / 100  # the y position for src top points
    top_y = img_h * 65 / 100  # the y position for src top points
    src_ps = np.float32(
        [[(img_w+top_dis)/2+x_top_right_offset, top_y], # top right
         [(img_w+bottom_dis)/2+x_bottom_right_offset, img_h], # bottom right
         [(img_w-bottom_dis)/2+x_bottom_left_offset, img_h],  # bottom left
         [(img_w-top_dis)/2 - 25, top_y]])  # top left
    dst_ps =  np.float32(
        [[img_w, 0], # top right
         [img_w, img_h], # bottom right
         [0, img_h],  # bottom left
         [0, 0]])  # top left
    return src_ps, dst_ps

## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--warp',
        default="test_images/test1.jpg",
        help='Warp the image')
    args = parser.parse_args()
    img = cv2.imread(args.warp)
    undistorted = camera_cal.undistort(img)
    src_ps, dst_ps = cal_warp_points(img)
    warped, M, _ = warp_image(undistorted, src_ps, dst_ps)
    show_two_images(img, warped,
                    src_ps, dst_ps,
                    title_src="Unditorted Image",
                    title_dst="Warped Image")