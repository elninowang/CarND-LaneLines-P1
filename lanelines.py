#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# #printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
# plt.show()

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    ysize = img.shape[0]
    xsize = img.shape[1]
    k_min = math.tan(30*math.pi/180)
    k_max = math.tan(60*math.pi/180)
    for line in lines:
        left_lines, right_lines = [],[]
        left_k_total, right_k_total = 0.0, 0.0,
        left_weight, right_weight = 0.0, 0.0
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1: break
                k = float(y2 - y1) / (x2 - x1)
                if abs(k) < k_min or abs(k) > k_max: break
                weight = ((y2-y1)**2 +(x2-x1)**2)**2
                if k < 0:
                    if x1 > xsize/2 or x2 > xsize/2: break
                    left_lines.append(line)
                    left_k_total += k * weight
                    left_weight += weight
                else:
                    if x1 < xsize / 2 or x2 < xsize / 2: break
                    right_lines.append(line)
                    right_k_total +=  k * weight
                    right_weight += weight
    if left_weight == 0 or right_weight == 0:
        return

    left_k = left_k_total/left_weight
    right_k = right_k_total/right_weight
    left_centers, right_centers = [],[]
    delta = math.pi*2/180
    for line in left_lines:
        for x1, y1, x2, y2 in line:
            k = float(y2 - y1) / (x2 - x1)
            if abs(math.atan(k) - math.atan(left_k)) < delta:
                left_centers.append([(x1+x2)/2,(y1+y2)/2, k])
                #cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 2)
    for line in right_lines:
        for x1, y1, x2, y2 in line:
            k = float(y2 - y1) / (x2 - x1)
            if abs(math.atan(k) - math.atan(right_k)) < delta:
                right_centers.append([(x1+x2)/2,(y1+y2)/2, k])
                #cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 2)
    y_front = 210
    if len(left_centers) > 0:
        left_center = np.mean(left_centers, axis=0)
        left_b = float(left_center[1]) - left_k * left_center[0]
        cv2.line(img, (int((ysize - left_b) / left_k), ysize), (int((ysize - y_front - left_b) / left_k), ysize - y_front),color, thickness)
    if len(right_centers) > 0:
        right_center = np.mean(right_centers, axis=0)
        right_b = float(right_center[1]) - right_k * right_center[0]
        cv2.line(img, (int((ysize-right_b)/right_k), ysize), (int((ysize-y_front-right_b)/right_k), ysize-y_front), color, thickness)

# def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., garma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + garma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, garma)

#TODO: Build your pipeline that will draw lane lines on the test_images
import os
image_files = os.listdir("test_images/")
for image_file in image_files:
    image = mpimg.imread("test_images/" + image_file)
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)

    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[(0,ysize),(xsize/2,310),(xsize,ysize)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    line_image = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=1, min_line_len=10, max_line_gap=1)
    combo = weighted_img(line_image, image)
    mpimg.imsave("test_images_output/" + image_file, combo)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# then save them to the test_images directory.
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)

    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[(0, ysize), (xsize/2, ysize/2), (xsize, ysize)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    line_image = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=1, min_line_len=10, max_line_gap=1)
    combo = weighted_img(line_image, image)
    return combo

white_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

#solidWhiteRight    solidYellowLeft     challenge
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)

white_clip.write_videofile(white_output, audio=False)