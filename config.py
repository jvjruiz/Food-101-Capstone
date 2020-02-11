import os

# initialize config variables
ROOT_IMAGE_PATH = '../data/images'

TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
VALID_DIR = "../data/valid"

IMG_SIZE = 224
IMG_HEIGHT = 224
IMG_WIDTH = 224

TRAIN_IMAGE_COUNT = 75750
VALID_IMAGE_COUNT = 12625
TEST_IMAGE_COUNT = 12625

MIN_LR = 1e-5
MAX_LR = 1e-1
BATCH_SIZE = 32
STEP_SIZE = 4
CLR_METHOD = 'triangular'
NUM_EPOCHS = 1000
EARLY_STOPPING = 10
