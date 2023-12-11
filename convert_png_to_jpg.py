import cv2
import os
from tqdm import tqdm
from pathlib import Path

dir_name = "/media/ssd1/pannuke/YOLO_train_data/images"

# traverse through all png files in the directory and resave them as jpg
for file_name in tqdm(Path(dir_name).glob("*.png"), desc="Converting png to jpg"):
    # read the png file
    image = cv2.imread(str(file_name))
    # get the stem of the file name
    stem = Path(file_name).stem
    # create the save path
    save_path = os.path.join(dir_name, stem + ".jpg")
    # save the image as jpg
    cv2.imwrite(save_path, image)

# delete all png files in the directory
for file_name in tqdm(Path(dir_name).glob("*.png"), desc="Deleting png files"):
    os.remove(file_name)
