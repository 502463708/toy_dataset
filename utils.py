import copy
import cv2
import numpy as np


def get_bounding_box_from_mask(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:  # and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def dilate_mask(src_label_np, radius):
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param radius:
    :return:
    """
    dilation_diameter = int(2 * radius + 1)
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [radius, radius])) <= radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.dilate(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np


def set_background_zero(src_image_np):
    num_value_0 = (src_image_np == 0).sum()
    num_value_non_0 = (src_image_np != 0).sum()
    dst_image_np = copy.copy(src_image_np)
    if num_value_non_0 > num_value_0:
        dst_image_np = 255 - dst_image_np

    return dst_image_np
