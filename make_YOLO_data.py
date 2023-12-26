import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from pathlib import Path


def get_bbox_from_mask(mask):
    # first get a list of all the x and y coordinates of the mask
    x_list = []
    y_list = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                x_list.append(j)
                y_list.append(i)

    TL_x = min(x_list)
    TL_y = min(y_list)
    BR_x = max(x_list)
    BR_y = max(y_list)

    return TL_x, TL_y, BR_x, BR_y


def get_separate_masks(label_path):
    # first open the annotation file which is .npy file, open as numpy array
    annotation = np.load(label_path)

    mask_list = []

    # get a list of all the unique values in the annotation
    unique_values = np.unique(annotation)

    # for each unique value, create a mask if the value is not 0
    for i in list(unique_values):
        if unique_values[i] != 0:
            mask = np.zeros(annotation.shape)
            mask[annotation == i] = 1
            mask_list.append(mask)

    return mask_list


def get_contour_points_from_mask(mask):
    """The mask is a 0 1 binary mask.
    We want to extract a list of boundary points from the mask.
    [(x1, y1), (x2, y2), ...] return a list of tuples.
    """

    # first convert the mask to uint8 ranging from 0 to 255
    mask = mask.astype(np.uint8)
    # change all 1 to 255
    mask[mask != 0] = 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Simplify contours and normalize points (example for the first contour)
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    normalized_contour = [
        (point[0][0] / mask.shape[1], point[0][1] / mask.shape[0]) for point in approx
    ]

    return normalized_contour


def get_txt_line_from_contour_points(contour_points):
    """Return a string where <class> is always 0
    <class> <x1> <y1> ... <xn> <yn>
    """
    txt_line = "0"

    for point in contour_points:
        txt_line += " " + str(point[0]) + " " + str(point[1])

    return txt_line


def get_boundary_points_txt_from_label(label_path, save_dir):
    """The label_path leads to a .npy file that contains the label of the image.
    The label path is in the format of /path/to/label_2136.npy.
    First get a list of masks from the the label.
    Then for each mask, get a list of boundary points.
    Fore each list of boundary points, convert it to a txt line.
    Write each line as a row in a txt file to save in the save_dir.
    The txt filename should be image_2136.txt.
    """

    mask_list = get_separate_masks(label_path)

    # get the stem of the label_path
    stem = Path(label_path).stem

    # the image name is image_2136.png, we want to save it as label_2136.jpg
    stem = stem.replace("label", "image")

    # create the save path
    save_path = os.path.join(save_dir, stem + ".txt")

    # open the save path as a file
    with open(save_path, "w") as f:
        # for each mask, get a list of boundary points, then convert it to a txt line, then write it to the file
        for i, mask in enumerate(mask_list):
            try:
                # get a list of boundary points from the mask
                contour_points = get_contour_points_from_mask(mask)

                # convert the list of boundary points to a txt line
                txt_line = get_txt_line_from_contour_points(contour_points)

                if i != len(mask_list) - 1:
                    # write the txt line to the file
                    f.write(txt_line + "\n")
                else:
                    f.write(txt_line)

            except Exception as e:
                print(e)
                print("Error in mask: ", mask)
                # print the dimension and the max min value of the mask
                print("Dimension: ", mask.shape)
                print("Max value: ", np.max(mask))
                print("Min value: ", np.min(mask))
                # display the mask, first convert to uint8 ranging from 0 to 255 (currently it is a 0 1 binary mask)
                mask = mask.astype(np.uint8)
                # change all 1 to 255
                mask[mask != 0] = 255
                cv2.imshow("mask", mask)
                cv2.waitKey(0)


def from_label_np_to_bbox_txt(label_path, save_dir):
    """The label_path leads to a .npy file that contains the label of the image.
    The label is a 256x256 binary mask with integer values.
    Each integer value corresponds to the mask of a different object.
    Find the bounding box of the mask of each object and save it as a csv file in the save_dir.
    """

    mask_list = get_separate_masks(label_path)

    df_rows = []

    for mask in mask_list:
        # display the mask, first convert to uint8 ranging from 0 to 255 (currently it is a 0 1 binary mask)
        run = False
        if run:
            mask = mask.astype(np.uint8)
            # change all 1 to 255
            mask[mask != 0] = 255
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # get the bounding box of the mask
        TL_x, TL_y, BR_x, BR_y = get_bbox_from_mask(mask)

        center = ((TL_x + BR_x) / 2, (TL_y + BR_y) / 2)

        center_x_rel = center[0] / 256
        center_y_rel = center[1] / 256

        width_rel = (BR_x - TL_x) / 256
        height_rel = (BR_y - TL_y) / 256

        dct = {
            "class": 0,
            "center_x_rel": center_x_rel,
            "center_y_rel": center_y_rel,
            "width_rel": width_rel,
            "height_rel": height_rel,
        }

        df_rows.append(dct)

    df = pd.DataFrame(df_rows)

    # get the stem of the label_path
    stem = Path(label_path).stem

    # create the save path
    save_path = os.path.join(save_dir, stem + ".txt")

    # save the dataframe as a txt file
    df.to_csv(save_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    label_dir = "/home/alpaca/Documents/neo/pannuke_full/labels/validation"
    save_dir = "/home/alpaca/Documents/neo/pannuke_full/YOLO_seg_data/validation/labels"

    # if the save_dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # traverse through all the label files in the label_dir
    for label_path in tqdm(Path(label_dir).glob("*.npy"), desc="Getting contours"):
        get_boundary_points_txt_from_label(label_path, save_dir)
