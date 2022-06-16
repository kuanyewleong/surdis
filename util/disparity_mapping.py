import numpy as np
import cv2 as cv
import tqdm
from matplotlib import pyplot as plt


# Load the left and right images in gray scale
imgL = cv.imread('sample_images/l_img_65.bmp', 0)
imgR = cv.imread('sample_images/r_img_65.bmp', 0)

# # Initialize the stereo block matching object
# stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)

# plt.imshow(disparity,'gray')
# plt.show()

def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    """
    Args:
        pixel_vals_1 (numpy.ndarray): pixel block from left image
        pixel_vals_2 (numpy.ndarray): pixel block from right image

    Returns:
        float: Sum of absolute difference between individual pixels
    """
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

BLOCK_SIZE = 9
SEARCH_BLOCK_SIZE = 16

def compare_blocks(y, x, block_left, right_array, block_size=5):
    """
    Compare left block of pixels with multiple blocks from the right
    image using SEARCH_BLOCK_SIZE to constrain the search in the right
    image.

    Args:
        y (int): row index of the left block
        x (int): column index of the left block
        block_left (numpy.ndarray): containing pixel values within the
                    block selected from the left image
        right_array (numpy.ndarray]): containing pixel values for the
                     entrire right image
        block_size (int, optional): Block of pixels width and height.
                                    Defaults to 5.

    Returns:
        tuple: (y, x) row and column index of the best matching block
                in the right image
    """
    # Get search range for the right image
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1], x + SEARCH_BLOCK_SIZE)
    #print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        sad = sum_of_abs_diff(block_left, block_right)
        #print(f'sad: {sad}, {y, x}')
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index

h, w = imgL.shape
disparity_map = np.zeros((h, w))
left_array = np.array(imgL)
right_array = np.array(imgR)

for y in range(BLOCK_SIZE, h-BLOCK_SIZE):
    for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
        block_left = left_array[y:y + BLOCK_SIZE,
                                x:x + BLOCK_SIZE]
        min_index = compare_blocks(y, x, block_left,
                                   right_array,
                                   block_size=BLOCK_SIZE)
        disparity_map[y, x] = abs(min_index[1] - x)

savePath = 'disparity_mapping/depth_in_npy_format/map.npy'
np.save(savePath, disparity_map)
# plt.imshow(disparity_map,'gray')
# plt.show()
