import os
import shutil
import zipfile
import json
import random
import argparse

from tqdm import tqdm

import cv2
import numpy as np

import torch

from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils


debug = False

unzip = False


def create_subdataset(images_path, labels_path, output_path, n=10):

    try:
        shutil.rmtree(output_path)
    except:
        pass
    finally:
        os.mkdir(output_path)
        os.mkdir(output_path + "/imgs")
        os.mkdir(output_path + "/bitmaps")
        os.mkdir(output_path + "/imgs/train")
        os.mkdir(output_path + "/bitmaps/train")
        os.mkdir(output_path + "/imgs/val")
        os.mkdir(output_path + "/bitmaps/val")
    print("Creating trainset...")
    for img in random.sample(os.listdir(images_path + "/train"), 5*n):
        shutil.copy(images_path + "/train/" + img, output_path + "/imgs/train")
        shutil.copy(labels_path + "/train/" + img[:-4] + "_drivable_color.png", output_path + "/bitmaps/train")
    print("Creating valset...")
    for img in random.sample(os.listdir(images_path + "/val"), n):
        shutil.copy(images_path + "/val/" + img, output_path + "/imgs/val")
        shutil.copy(labels_path + "/val/" + img[:-4] + "_drivable_color.png", output_path + "/bitmaps/val")


def unzip_bdd100k(output_path, zip_path):
    with zipfile.ZipFile(zip_path, "r") as zipPointer:
        zipPointer.extractall(output_path)


def create_annotations(img, filename, idx, images_path, pad=1):
    # Bigger value means slower to find the class of the object
    class_threshold = 75

    # Add blur to smooth edges
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Erode the bitmap to create a gap between annotations
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=4)

    # Detect edges
    edged = cv2.Canny(img, 30, 200)

    # Do a morphologic close to connect edges
    kernel = np.ones((3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Return dict
    record = {}

    record["file_name"] = images_path + "/" + filename
    record["image_id"] = idx
    record["height"] = img.shape[0] - 2 * pad
    record["width"] = img.shape[1] - 2 * pad

    objs = []
    for obj in contours:

        np_obj = np.asarray(obj)

        # Get BBox corners
        np_obj = np_obj.squeeze(1)
        max_x = np_obj[np_obj[:, 0].argmax()][0]
        min_x = np_obj[np_obj[:, 0].argmin()][0]

        max_y = np_obj[np_obj[:, 1].argmax()][1]
        min_y = np_obj[np_obj[:, 1].argmin()][1]

        # Object
        obj = img[min_y:max_y, min_x:max_x]

        # Find class id
        # class_id = np.mean(obj, axis=(0, 1)).argmax()

        count = 0
        tries = 0

        # Generate random points and find the color of the objects bitmap
        pixel_sum = [0, 0, 0]
        while count < 5:
            y = random.randint(min_y, max_y)
            x = random.randint(min_x, max_x)

            if cv2.pointPolygonTest(np_obj, (x, y), False) == 1:
                count += 1
                pixel_sum += img[y, x]

            tries += 1

            if tries > class_threshold:
                break

        # If could not find a pixel in the contour, discard it
        if tries > class_threshold:
            continue
        else:
            class_id = pixel_sum.argmax()

        if debug:
            # Draw bboxes to the dummy image for debug purposes
            img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Remove the custom pad and add 0.5 standard pad the framework wants.
        poly = [p - pad + 0.5 for x in np_obj for p in x]

        obj = {
            "bbox": [min_x - pad, min_y - pad, max_x - pad, max_y - pad],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": class_id
        }

        objs.append(obj)

    record["annotations"] = objs

    if debug:
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return record


def get_dataset_dicts(images_path, bitmap_path):

    pad = 10

    img_list = os.listdir(bitmap_path)

    dataset_dicts = []

    for idx, bitmap in tqdm(enumerate(img_list)):
        img = read_img(bitmap_path + "/{}".format(bitmap), pad=pad)
        file_name = bitmap.split("_")[0] + ".jpg"
        dataset_dicts.append(create_annotations(img, file_name, idx, images_path, pad=pad))

    return dataset_dicts


def read_img(img_path, pad=1):
    bgr_img = cv2.imread(img_path)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    padded_image = cv2.copyMakeBorder(rgb_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    return padded_image


def mapper(sample):
    image = utils.read_image(sample["file_name"], format="BGR")

    image = image.copy()

    # apply data augmentation here

    auginput = image

    image = torch.from_numpy(auginput.transpose(2, 0, 1))
    annos = [
        annotation
        for annotation in sample.pop("annotations")
    ]
    return {
        # create the format that the model expects
        "image": image,
        "instances": utils.annotations_to_instances(annos, image.shape[1:], mask_format="polygon")
    }