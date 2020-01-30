import os

# initialize config variables
ROOT_IMAGE_PATH = '../data/images'

TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
VALID_DIR = "../data/valid"
SUBSET_TRAIN_DIR = "../data/subset/train"
SUBSET_TEST_DIR = "../data/subset/test"
SUBSET_VALID_DIR = "../data/subset/valid"
RESIZED_SUBSET_TRAIN_DIR = "../data/subset/resized/train"
RESIZED_SUBSET_TEST_DIR = "../data/subset/resized/test"
RESIZED_SUBSET_VALID_DIR = "../data/subset/resized/valid"
RESIZED_TRAIN_DIR = "../data/resized/train"
RESIZED_TEST_DIR = "../data/resized/test"
RESIZED_VALID_DIR = "../data/resized/valid"
FINAL_SIZE = 224
IMG_HEIGHT = 299
IMG_WIDTH = 299

TRAIN_IMAGE_COUNT = 75750
VALID_IMAGE_COUNT = 12625
TEST_IMAGE_COUNT = 12625

SUBSET_TRAIN_IMAGE_COUNT = 40400
SUBSET_VALID_IMAGE_COUNT = 5050
SUBSET_TEST_IMAGE_COUNT = 5050

MIN_LR = 1e-5
MAX_LR = 1e-1
BATCH_SIZE = 32
STEP_SIZE = 4
CLR_METHOD = 'triangular'
NUM_EPOCHS = 1000
EARLY_STOPPING = 10

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])

SERVER_URL = 'http://localhost:8501'