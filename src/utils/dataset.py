# import necessary packages
import numpy as np

import os, sys
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict
from PIL import Image
import pandas as pd


from glob import glob

import config as cfg

# create functions for copying files and ignoring files

def copytree(src, dst, ids_to_copy = None):
    # src = source directory
    # dst = destination directory of copy
    # if destination directory does not exist, create directory
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    
    # get list of directories in current directory
    directory_items = os.listdir(src)

    # for each item in directory, copy into destination
    for item in directory_items:
        source = os.path.join(src, item)
        destination = os.path.join(dst, item)
        # if item is a directory, recurisvely call this function 
        if os.path.isdir(source):
            print(source)
            copytree(source, destination, ids_to_copy)
        # copy item to destination

        elif item in ids_to_copy:
            shutil.copy2(source, destination)

# generate list of folders and filenames in directory from metadata txt file
def generate_dir_file_map(path):
    dir_files = defaultdict(list)
    with open(path, 'r') as txt:
        files = [l.strip() for l in txt.readlines()]
        for f in files:
            dir_name, id = f.split('/')
            dir_files[dir_name].append(id + '.jpg')
    return dir_files

# generate list of id's of training images
def generate_training_ids():
    train_dir_files = generate_dir_file_map('../data/meta/train.txt')
    ids_to_copy = list()
    for category in train_dir_files:
        for file in train_dir_files[category][:400]:
            ids_to_copy.append(file)
    return ids_to_copy

# generate list of id's of using the first half of the testing images
def generate_testing_ids():
    test_dir_categories = generate_dir_file_map('../data/meta/test.txt')
    ids_to_copy = list()
    for category in test_dir_categories:
        # first_half_of_test_ids = test_dir_categories[category][:len(test_dir_categories)//2]
        for image_id in test_dir_categories[category][40:80]:
            ids_to_copy.append(image_id)
    return ids_to_copy

# generate list of id's of using the second half of the testing images
def generate_validation_ids():
    test_dir_categories = generate_dir_file_map('../data/meta/test.txt')
    ids_to_copy = list()
    for category in test_dir_categories:
        # second_half_of_test_ids = test_dir_categories[category][len(test_dir_categories//2):]
        for image_id in test_dir_categories[category][:120]:
            ids_to_copy.append(image_id)
    return ids_to_copy

# takes images from original directory and splits them into train/test/valid directories
def sort_images():
    if not os.path.isdir(cfg.SUBSET_TRAIN_DIR):
        copytree(cfg.ROOT_IMAGE_PATH, cfg.SUBSET_TRAIN_DIR, ids_to_copy=generate_training_ids())
    else:
        print('Train files already copied into separate folders.')

    # if not os.path.isdir(cfg.SUBSET_TEST_DIR):
    #     copytree(cfg.ROOT_IMAGE_PATH, cfg.SUBSET_TEST_DIR, ids_to_copy=generate_testing_ids())
    # else:
    #     print('Test files already copied into separate folders.')

    if not os.path.isdir(cfg.SUBSET_VALID_DIR):
        copytree(cfg.ROOT_IMAGE_PATH, cfg.SUBSET_VALID_DIR, ids_to_copy=generate_validation_ids())
    else:
        print('Validation files already copied into separate folders.')

# take root path and label name of training or test directories and makes copies at a standard pixel size
def resize_images(path, directory,category, width, height):
    images = os.listdir("{}/{}".format(path,directory))
    for item in images:
        if item == '.DS_Store':
             continue
        if os.path.isfile("{}/{}/{}".format(path,directory,item)):
            save_path = "../data/subset/resized/{}/{}/{}".format(category, directory, item)
            im = Image.open("{}/{}/{}".format(path,directory,item))

            resized_im = resize_by_enlarge(im, width, height)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            resized_im.save(save_path, 'JPEG', quality=90)

    print("Finished resizing images from {} {} set".format(directory, category))

# resize images by finding the smaller dimension and enlarging 
# height and width by the ratio of the final size to smaller dimension
def resize_by_enlarge(im, width, height):
    w, h = im.size
    final_size = cfg.FINAL_SIZE
    new_image_size = tuple([w,h])
    if w < final_size:
        # get ratio of final size to width
        ratio = final_size/float(w)
        # increase height by ratio
        new_image_size = tuple([int(final_size),int(h*ratio)])
        # print('resized item {}'.format(item))
        print(new_image_size)
    elif h < final_size:
        ratio = final_size/float(h)
        new_image_size = tuple([int(w * ratio), int(final_size)])
        # print('resized item {}'.format(item))
        print(new_image_size)
    resized_im = im.resize(new_image_size)
    return resized_im



# resizes images to specified height/width while keeping a consistent ratio
# by filling the rest of the picture with black
def resize_image_and_keep_aspect(im, final_width, final_height):
    size = im.size
    ratio = float(cfg.FINAL_SIZE) / max(size)
    resized_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(resized_image_size, Image.ANTIALIAS)
    resized_im = Image.new("RGB", (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    resized_im.paste(im, ((cfg.IMG_WIDTH-resized_image_size[0])//2, (cfg.IMG_HEIGHT-resized_image_size[1])//2))

    return resized_im
    
def return_resized_dir(category):
    print(category)
    if category == 'train':
        return cfg.RESIZED_TRAIN_DIR
    elif category == 'valid':
        return cfg.RESIZED_VALID_DIR
    elif category == 'test':
        return cfg.RESIZED_TEST_DIR

def get_labels():
    food_labels = pd.read_csv("../../data/meta/labels.txt", header=None)
    food_labels = food_labels[0].tolist()
    return food_labels