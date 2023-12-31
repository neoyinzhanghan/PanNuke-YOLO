import cv2
import os
from tqdm import tqdm
from pathlib import Path

dir_name = "/home/alpaca/Documents/neovision/pannuke_full/YOLO_seg_data/train/images"

# traverse through all png files in the directory and resave them as jpg
for file_name in tqdm(Path(dir_name).glob("*.png"), desc="Converting png to jpg"):
    # read the png file
    image = cv2.imread(str(file_name))
    # get the stem of the file name
    stem = Path(file_name).stem

    # the image name is image_2136.png, we want to save it as label_2136.jpg
    stem = stem.replace("image", "label")

    # create the save path
    save_path = os.path.join(dir_name, stem + ".jpg")
    # save the image as jpg
    cv2.imwrite(save_path, image)

# delete all png files in the directory
for file_name in tqdm(Path(dir_name).glob("*.png"), desc="Deleting png files"):
    os.remove(file_name)
