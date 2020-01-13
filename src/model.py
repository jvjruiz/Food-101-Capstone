# load the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import Model

from utils.model import euclidean_distance_loss  

from config import *

def build_and_train_model():
    # get base model for transfer learning, in this case VGG16
    base_model = get_base_model()

    #add in final layers of model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(4,4))(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(final_size, activation='tanh')(head_model)
    head_model = Dropout(0.4)(head_model)
    head_model = Dense(101, activation='softmax')(head_model)

    #place new head model on top of pre-trained network
    model = Model(inputs=base_model.input, outputs=head_model)

    # compile model
    model.compile(optimizer='adam',
                loss=euclidean_distance_loss,
                metrics=['accuracy'])

    # initialize early stopping callback
    es_callback = EarlyStopping(monitor="val_loss", patience=early_stopping_patience)
    csv_logger = CSVLogger('../history/VGG16_v3.log')

    train_data_gen, valid_data_gen = get_image_generators()

    # train model
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_image_count//batch_size,
        epochs=epochs,
        validation_data=valid_data_gen,
        validation_steps=valid_image_count//batch_size,
        workers=4,
        use_multiprocessing=False,
        callbacks=[es_callback, csv_logger]
    )

    # if there is not a saved model, save model for future use
    if not os.path.isdir('../saved_models'):
        os.makedirs('../saved_models')
    model.save('../saved_models/CNN_VGG16_MODEL_V2.h5')

    return model

def get_base_model():
    # import VGG16 base model for transfer learning
    # freeze weights for dense layers at the top so we can add our own
    base_model = VGG16(weights='imagenet', 
              include_top=False, 
              input_tensor=(Input(shape=(IMG_HEIGHT,IMG_WIDTH,3)))
             )

    # freeze the top layers of the model so they don't train
    for layer in base_model.layers:
        layer.trainable = False
    
    return base_model

def get_image_generators():
    train_image_generator = ImageDataGenerator(rescale=1./255)
    valid_image_generator = ImageDataGenerator(rescale=1./255)

    # initialize training image iterator
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=resized_train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT,IMG_WIDTH))

        # initialize test image iterator
    valid_data_gen = valid_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=resized_valid_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT,IMG_WIDTH))
                                                    
    return train_data_gen, valid_data_gen

build_and_train_model()