# load the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import clear_session

import numpy as np
import argparse

from utils.clr_callback import CyclicLR
from utils.learning_rate_finder import LearningRateFinder
from utils.model import euclidean_distance_loss, random_crop, crop_generator, get_image_generators, get_base_model  
import config as cfg

import matplotlib.pyplot as plt

import sys
import pdb
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
ap.add_argument("-s", "--subset", type=int, default=0,
    help="whether or not to use subset of data")
args = vars(ap.parse_args())

def build_and_train_model():
    clear_session()
    # get base model for transfer learning, in this case VGG16
    base_model = get_base_model()

    #add in final layers of model
    head_model = base_model.output
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(cfg.FINAL_SIZE, activation='relu')(head_model)
    head_model = Dropout(0.4)(head_model)
    head_model = Dense(101, activation='softmax')(head_model)

    #place new head model on top of pre-trained network
    model = Model(inputs=base_model.input, outputs=head_model)

    opt = SGD(lr=cfg.MIN_LR, momentum=0.9)

    #     # freeze the top layers of the model so they don't train
    # for layer in base_model.layers:
    #     layer.trainable = False

    # compile model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    train_dir = cfg.RESIZED_TRAIN_DIR
    valid_dir = cfg.RESIZED_VALID_DIR

    train_image_count = cfg.TRAIN_IMAGE_COUNT
    valid_image_count = cfg.VALID_IMAGE_COUNT

    if args['subset'] > 0:
        train_dir = cfg.RESIZED_SUBSET_TRAIN_DIR
        valid_dir = cfg.RESIZED_SUBSET_VALID_DIR
        train_image_count = cfg.SUBSET_TRAIN_IMAGE_COUNT
        valid_image_count = cfg.SUBSET_VALID_IMAGE_COUNT

    train_data_gen, valid_data_gen = get_image_generators(train_dir, valid_dir)

    cropped_train_data_gen = crop_generator(train_data_gen, cfg.FINAL_SIZE)
    cropped_valid_data_gen = crop_generator(valid_data_gen, cfg.FINAL_SIZE)

    # check to see if we are attempting to find an optimal learning rate
    # before training for the full number of epochs
    if args["lr_find"] > 0:
        # initialize the learning rate finder and then train with learning
        # rates ranging from 1e-10 to 1e+1
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find(
            cropped_train_data_gen,
            1e-7, 1,
            steps_per_epoch=np.ceil(train_image_count / float(cfg.BATCH_SIZE)),
            batch_size=cfg.BATCH_SIZE,
            epochs=30)
 
        # plot the loss for the various learning rates and save the
        # resulting plot to disk
        lrf.plot_loss()
        plt.savefig(cfg.LRFIND_PLOT_PATH)
    
        # gracefully exit the script so we can adjust our learning rates
        # in the config and then train the network for our full set of
        # epochs
        print("[INFO] learning rate finder complete")
        print("[INFO] examine plot and adjust learning rates before training")
        sys.exit(0)

    # initialize early stopping callback
    es_callback = EarlyStopping(monitor="val_loss", patience=cfg.EARLY_STOPPING)
    csv_logger = CSVLogger('../history/VGG16_v5.log')


    valid_step_size = valid_image_count//cfg.BATCH_SIZE
    train_step_size= train_image_count//cfg.BATCH_SIZE

    clr = CyclicLR(
        mode=cfg.CLR_METHOD,
        base_lr=cfg.MIN_LR,
        max_lr=cfg.MAX_LR,
        step_size=train_step_size * cfg.STEP_SIZE
    )

    checkpointer = ModelCheckpoint(filepath='../checkpoints/modelV5b.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)

    # train model
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_image_count//cfg.BATCH_SIZE,
        epochs=cfg.NUM_EPOCHS,
        validation_data=valid_data_gen,
        validation_steps=valid_image_count//cfg.BATCH_SIZE,
        callbacks=[es_callback, csv_logger, checkpointer]
    )

    # if there is not a saved model, save model for future use
    if not os.path.isdir('../saved_models'):
        os.makedirs('../saved_models')
    model.save('../saved_models/CNN_VGG16_MODEL_V5.h5')

    return model

build_and_train_model()