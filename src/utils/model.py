import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input

import numpy as np

import config as cfg
import pdb

def get_base_model():
    # import VGG16 base model for transfer learning
    # freeze weights for dense layers at the top so we can add our own
    base_model = VGG16(weights='imagenet', 
              include_top=False, 
              input_tensor=(Input(shape=(cfg.FINAL_SIZE,cfg.FINAL_SIZE,3)))
             )


    
    return base_model

def get_image_generators(train_dir, valid_dir):
    # train_image_generator = ImageDataGenerator(rescale=1./255,
    #                                             horizontal_flip=True,
    #                                             zoom_range=[.8,1],
    #                                             fill_mode='reflect',
    #                                             width_shift_range=0.2,
    #                                             height_shift_range=0.2)
    train_image_generator = ImageDataGenerator(rescale=1./255)
    valid_image_generator = ImageDataGenerator(rescale=1./255)
    # initialize training image iterator
    train_data_gen = train_image_generator.flow_from_directory(batch_size=cfg.BATCH_SIZE,
                                                            directory=train_dir,
                                                            seed=42,
                                                            shuffle=True,
                                                            target_size=(cfg.FINAL_SIZE, cfg.FINAL_SIZE)
                                                            )

        # initialize test image iterator
    valid_data_gen = valid_image_generator.flow_from_directory(batch_size=cfg.BATCH_SIZE,
                                                            directory=valid_dir,
                                                            seed=42,
                                                            target_size=(cfg.FINAL_SIZE, cfg.FINAL_SIZE),
                                                            shuffle=True)
                                                    
    return train_data_gen, valid_data_gen

def crop_generator(batches, target_dimension):
    # take a keras image generator and generates random crops
    # from the image batches
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], target_dimension, target_dimension, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (target_dimension, target_dimension))
        yield (batch_crops, batch_y)

def random_crop(img, crop_dimensions):
    # takes image tensor data and randomly takes a crop with provided dimensions
    assert img.shape[2] == 3
    h, w = img.shape[0], img.shape[1]
    dy, dx = crop_dimensions
    x = np.random.randint(0, w - dx + 1)
    y = np.random.randint(0, h - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))



