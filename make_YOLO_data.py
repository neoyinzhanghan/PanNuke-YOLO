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
    df.to_csv(save_path, sep=" ", index=False, header=False)


if __name__ == "__main__":
    label_dir = "/Users/neo/Documents/Research/CP/pannuke/labels/masks/validation"
    save_dir = "/Users/neo/Documents/Research/CP/pannuke/bboxes/masks/validation"

    # if save_dir does not exist, create it
    os.makedirs(save_dir, exist_ok=True)

    # traverse through all the .npy files in the label_dir
    for label_path in tqdm(
        Path(label_dir).glob("*.npy"), desc="Creating bbox txt files"
    ):
        # convert the label to bbox and save it as a txt file
        from_label_np_to_bbox_txt(label_path, save_dir)
