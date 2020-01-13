import os

# initialize config variables
root_image_path = '../data/images'

train_dir = "../data/train"
test_dir = "../data/test"
valid_dir = "../data/valid"
resized_train_dir = "../data/resized/train"
resized_test_dir = "../data/resized/test"
resized_valid_dir = "../data/resized/valid"
final_size = 256
IMG_HEIGHT = 256
IMG_WIDTH = 256
train_image_count = 75750
valid_image_count = 12625
test_image_count = 12625
epochs = 1000
early_stopping_patience = 10
batch_size = 64 # change with pc specs



SERVER_URL = 'http://localhost:8501'