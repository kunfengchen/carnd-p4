import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

### REF: ported from the class material

def color_pipeline(img,
                   bin_output=True,
                   s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Apply Sobel in x and HLS pipeline
    :param img:
    :param bin_output: bin_image output or not
    :param s_thresh: Threshold for color channel
    :param sx_thresh: Threshold for x gradient
    :return: if bin_output, one-channel, three-channel otherwise
    """
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    if bin_output:
        bin_out = np.zeros_like(s_channel)
        bin_out[
            (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) |
            (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        return bin_out
    else:
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary

def show_threshold_images(img, thresh_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(thresh_img, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--colored',
        default="test_images/test1.jpg",
        help='Apply Sobel to the image')
    args = parser.parse_args()
    img = cv2.imread(args.colored)
    thresh_img = color_pipeline(img, bin_output=False)
    # Plot the result
    show_threshold_images(img, thresh_img)
