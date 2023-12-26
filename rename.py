import os
from pathlib import Path
from tqdm import tqdm

dir_path = "/home/alpaca/Documents/neo/pannuke_full/YOLO_seg_data/train/images"

# traverse through all jpg files in the directory
# the file names are labels_XXX.jpg, we want to rename them to image_XXX.jpg
for file_name in tqdm(Path(dir_path).glob("*.jpg"), desc="Renaming files"):
    # get the stem of the file name
    stem = Path(file_name).stem

    # the image name is label_2136.jpg, we want to save it as image_2136.jpg
    stem = stem.replace("label", "image")

    # create the save path
    save_path = os.path.join(dir_path, stem + ".jpg")

    # rename the file
    os.rename(file_name, save_path)
