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

# create functions for copying files and ignoring files

def copytree(src, dst, ignored_ids = None):
    # src = source directory
    # dst = destination directory of copy
    # ignore = ignore function that provides list of id's to ignore based on testing or training set
    
    # if destination directory does not exist, create directory
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    
    # get list of directories in current directory
    directory_items = os.listdir(src)
    # filter out items to be ignored
    directory_items = [x for x in directory_items if x not in ignored_ids]
    # for each item in directory, copy into destination
    for item in directory_items:
        source = os.path.join(src, item)
        destination = os.path.join(dst, item)
        # if item is a directory, recurisvely call this function 
        if os.path.isdir(source):
            print(source)
            copytree(source, destination, ignored_ids)
        # copy item to destination
        else:
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
    ids_to_ignore = list()
    for category in train_dir_files:
        for file in train_dir_files[category]:
            ids_to_ignore.append(file)
    return ids_to_ignore

# generate list of id's of using the first half of the testing images
def generate_testing_ids():
    test_dir_categories = generate_dir_file_map('../data/meta/test.txt')
    ids_to_ignore = list()
    for category in test_dir_categories:
        first_half_of_test_ids = test_dir_categories[category][:len(test_dir_categories[category])//2]
        for image_id in first_half_of_test_ids:
            ids_to_ignore.append(image_id)
    return ids_to_ignore

# generate list of id's of using the second half of the testing images
def generate_validation_ids():
    test_dir_categories = generate_dir_file_map('../data/meta/test.txt')
    ids_to_ignore = list()
    for category in test_dir_categories:
        second_half_of_test_ids = test_dir_categories[category][len(test_dir_categories[category])//2:]
        for image_id in second_half_of_test_ids:
            ids_to_ignore.append(image_id)
    return ids_to_ignore

# takes images from original directory and splits them into train/test/valid directories
def sort_images():
    # Only split files if haven't already
    ids_to_ignore_for_training = generate_testing_ids() + generate_validation_ids()
    ids_to_ignore_for_testing = generate_training_ids() + generate_validation_ids()
    ids_to_ignore_for_validation = generate_testing_ids() + generate_training_ids()
    
    root_dir = '../data/images'
    train_dir = '../data/train'
    test_dir = '../data/test'
    valid_dir = '../data/valid'
    
    if not os.path.isdir('../data/test') and not os.path.isdir('../data/train'):
        copytree(root_dir, train_dir, ignored_ids=ids_to_ignore_for_training)
    else:
        print('Train files already copied into separate folders.')

    if not os.path.isdir('../data/test'):
        copytree(root_dir, test_dir, ignored_ids=ids_to_ignore_for_testing)
    else:
        print('Test files already copied into separate folders.')

    if not os.path.isdir('../data/valid'):
        copytree(root_dir, valid_dir, ignored_ids=ids_to_ignore_for_validation)
    else:
        print('Validation files already copied into separate folders.')

# take root path and label name of training or test directories and makes copies at a standard pixel size
def resize_aspect_fit_images(path, directory,category, height, width):
    images = os.listdir("{}/{}".format(path,directory))
    for item in images:
        if item == '.DS_Store':
             continue
        if os.path.isfile("{}/{}/{}".format(path,directory,item)):
            save_path = "../data/resized/{}/{}/{}".format(category, directory, item)
            im = Image.open("{}/{}/{}".format(path,directory,item))

            resized_im = resize_image(im)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            resized_im.save(save_path, 'JPEG', quality=90)

    print("Finished resizing images from {} {} set".format(directory, category))

# needs PIL image
def resize_image(im):
    print(im.size)
    size = im.size
    ratio = float(256) / max(size)
    resized_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(resized_image_size, Image.ANTIALIAS)
    resized_im = Image.new("RGB", (256, 256))
    resized_im.paste(im, ((256-resized_image_size[0])//2, (256-resized_image_size[1])//2))

    return resized_im

def get_labels():
    food_labels = pd.read_csv("../../data/meta/labels.txt", header=None)
    food_labels = food_labels[0].tolist()
    return food_labels