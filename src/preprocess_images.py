from utils.dataset import *
import config as cfg 

#function that takes raw image data, sorts, and resizes into respective training/test/validation folders
def preprocess_images():
    sort_images()
    if not os.path.isdir(cfg.RESIZED_SUBSET_TRAIN_DIR):
        for directory in os.listdir(cfg.SUBSET_TRAIN_DIR):
            resize_images(cfg.SUBSET_TRAIN_DIR,directory,'train',cfg.IMG_WIDTH,cfg.IMG_HEIGHT)
    else:
        print('Train files already resized')
    if not os.path.isdir(cfg.RESIZED_SUBSET_TEST_DIR):
        for directory in os.listdir(cfg.SUBSET_TEST_DIR):
            resize_images(cfg.SUBSET_TEST_DIR,directory,'test',cfg.IMG_WIDTH,cfg.IMG_HEIGHT)
    else:
        print('Test files already resized')
    if not os.path.isdir(cfg.RESIZED_SUBSET_VALID_DIR):
        for directory in os.listdir(cfg.SUBSET_VALID_DIR):
            resize_images(cfg.SUBSET_VALID_DIR, directory, 'valid',cfg.IMG_WIDTH,cfg.IMG_HEIGHT)
    else:
        print('Valid files already resized')

preprocess_images()
