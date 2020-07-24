"""
Download, extract and split example slide images.
"""

import csv
import os
import urllib.request
import zipfile
from shutil import copyfile

from PIL import Image
from tqdm import tqdm
import numpy as np

from util import create_dir_if_not_exist


DATA_RAW_DIR = "./dataset"
# EXAMPLE_SLIDES_ZIP = DATA_RAW_DIR + "/example_slides.zip"
IOSTAR_IMAGE = DATA_RAW_DIR + "/valid"
IOSTAR_GT = DATA_RAW_DIR + "/valid_GT"

PROCESSED_IOSTAR_DIR_IMAGE = "./processed/48/valid"
PROCESSED_IOSTAR_DIR_GT = "./processed/48/valid_GT"


def create_patch(whole_slide_dir, patch_dir, patch_size):
    # Create dirs
    responder_dir = patch_dir + "/1st_manual"
    non_responder_dir = patch_dir
    # create_dir_if_not_exist(responder_dir)
    create_dir_if_not_exist(non_responder_dir)
    create_dir_if_not_exist("processed")


    # Iterate through files to split and group them
    image_files = os.listdir(whole_slide_dir)
    print(len(image_files), "slide images found")
    total = 0
    skipped = []
    for image_file in tqdm(image_files, desc="Splitting images"):
        if "DS_Store" not in image_file:
            image = Image.open(whole_slide_dir + "/" + image_file)
            width, height = image.size
            file_well_num = image_file[:image_file.rindex(".")]

            save_dir = responder_dir if "1st_manual" in image_file else non_responder_dir

            # Round to lowest multiple of target width and height.
            # Will lead to a loss of image data around the edges, but ensures split images are all the same size.
            rounded_width = patch_size * (width // patch_size)
            rounded_height = patch_size * (height // patch_size)

            # Split and save
            xs = range(0, rounded_width, patch_size)
            ys = range(0, rounded_height, patch_size)
            for i_x, x in enumerate(xs):
                for i_y, y in enumerate(ys):
                    box = (x, y, x + patch_size, y + patch_size)
                    cropped_data = image.crop(box)
                    # print(cropped_data)
                    cropped_image = Image.new('RGB', (patch_size, patch_size), 255)
                    cropped_image.paste(cropped_data)
                    np_data = np.array(cropped_image)
                    # print(np_data.shape)
                    if np.mean(np_data[:, :, :1]) == 0:
                        continue

                    processed_GT_file = os.listdir("processed/48/valid_GT")

                    if "GT" in whole_slide_dir:
                        cropped_image.save(save_dir + "/" + file_well_num + "_x" + str(i_x) + "_y" + str(i_y) + ".png")
                    else:
                        naming_string = file_well_num + "_x" + str(i_x) + "_y" + str(i_y) + ".png"
                        if naming_string not in processed_GT_file:
                            continue
                        cropped_image.save(save_dir + "/" + file_well_num + "_x" + str(i_x) + "_y" + str(i_y) + ".png")
                    total += 1

    print('Created', total, 'split images')
    if skipped:
        print('Labels not found for', skipped, 'so they were skipped')


if __name__ == "__main__":
    patch_size = 48
    print('===================== splitting GT ====================================')
    create_patch(IOSTAR_GT, PROCESSED_IOSTAR_DIR_GT, patch_size)

    print('===================== splitting images ====================================')
    create_patch(IOSTAR_IMAGE, PROCESSED_IOSTAR_DIR_IMAGE, patch_size)

    processed_GT = os.listdir("processed/48/valid_GT")
    processed_IMAGE = os.listdir("processed/48/valid")

    missing = []
    nk = set(processed_IMAGE).intersection(processed_GT)
    for x in processed_IMAGE:
        if x in nk:
            continue
        missing.append(x)

    print(len(missing))
    print(missing)


