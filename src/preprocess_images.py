from utils.dataset import *
from config import *

#function that takes raw image data, sorts, and resizes into respective training/test/validation folders
def preprocess_images():
    sort_images()
    if not os.path.isdir(resized_train_dir):
        for directory in os.listdir(train_dir):
            resize_aspect_fit_images(train_path,directory,'train',final_size,final_size)
    else:
        print('Train files already resized')
    if not os.path.isdir(resized_test_dir):
        for directory in os.listdir(test_dir):
            resize_aspect_fit_images(test_path,directory,'test',final_size,final_size)
    else:
        print('Train files already resized')
    if not os.path.isdir(resized_valid_dir):
        for directory in os.listdir(valid_dir):
            resize_aspect_fit_images(valid_path, directory, 'valid',final_size, final_size)
    else:
        print('Valid files already resized')

preprocess_images()
