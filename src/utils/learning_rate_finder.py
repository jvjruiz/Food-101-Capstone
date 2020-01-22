# import necessary packages
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

import config as cfg

class LearningRateFinder:
    def __init__(self, model, stop_factor = 4, beta = 0.98):
        # store the model, stop factor, and beta value ( for computing a smoothed, average loss)
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta

        # initialize our list of learning rates and losses,
        self.lrs = []
        self.losses = []

        # initialize out learning rate multiplier, average loss, best
        # loss found thus found, current batch number, and weights file
        self.lr_multiplier = 1
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num= 0
        self.weights_file = None

    def reset(self):
        self.lr_multiplier = 0
        self.avg_loss = 0
        self.best_loss = 1e9
        self.batch_num= 0
        self.weights_file = None

    def is_data_generator(self, data):
        # define the set of class types we will check for
        iter_classes = ['NumpyArrayIterator', 'DirectoryIterator',
        'DataFrameIterator', 'Iterator', 'Sequence', 'generator']

        # return whether out data is an iterator
        return data.__class__.__name__ in iter_classes
    
    def on_batch_end(self, batch, logs):
        # grab the current learning rate and log it to the list
        # of learning rates we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end of this bastch, increment the total
        # number of batch processed, compute the average
        # loss, smooth it, and update the losses list with the smoothed value
        l = logs['loss']
        self.batch_num += 1
        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * l)
        smooth = self.avg_loss / (1 -(self.beta ** self.batch_num))
        self.losses.append(smooth)

        # compute the maximum loss topping factor value
        stop_loss = self.stop_factor * self.best_loss

        # check to see if stop loss has grown too larger
        if self.batch_num > 1 and smooth > stop_loss:
            # stop training and return from the method
            self.model.stop_training = True
            return

        # check to see if best loss should be updated
        if self.batch_num == 1 or smooth < self.best_loss:
            self.best_loss = smooth

        # increase the learning rate
        lr *= self.lr_multiplier
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, train_data, start_LR, end_LR, epochs=None,
        steps_per_epoch=None, batch_size=64, sample_size=75750,
        verbose=1):
        # reset out class-specific variables
        self.reset()

        # determine if we are using a data generator or not
        use_gen = self.is_data_generator(train_data)

        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if use_gen and steps_per_epoch is None:
            msg = "Using generator without supplying steps_per_epoch"
            raise Exception(msg)
        # if we're not using a generator then our entire dataset must
        # already be in memory
        elif not use_gen:
            # grab the number of samples in the training data and
            # then derive the number of steps per epoch
            num_samples = len(train_data[0])
            steps_per_epoch = np.ceil(num_samples / float(batch_size))

        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sample_size / float(steps_per_epoch)))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        num_batch_updates = epochs * steps_per_epoch

        # derive the learning rate multiplier based on the ending
        # learning rate, starting leraning rate, and total number of
        # batch updates
        self.lr_multiplier = (end_LR / start_LR) ** (1.0 / num_batch_updates)

        # create a temp file path for the model weights and
        # then save the weights ( so we can reset the weights when
        # we are done)
        self.weights_file = tempfile.mkstemp()[1]
        self.model.save_weights(self.weights_file)

        #grab the *original* learning rate( so we can reset it
        # later), and the set the *starting* learning rate
        orig_LR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_LR)

        # construct a callback that will be caleld at the end of each
        # batch, enabling us to increase out leranring rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch,logs))

        # check to see if we are using a data iterator
        if use_gen:
            self.model.fit_generator(
                train_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback]
            )

        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                train_data[0], train_data[1],
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose)

        # restore the original model weights and learning rate
        self.model.load_weights(self.weights_file)
        K.set_value(self.model.optimizer.lr, orig_LR)

    def plot_loss(self, skip_begin=10, skip_end=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]

        # plot the learning rate vs loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning rate (Log Scale)")
        plt.ylabel("loss")

        # if the title is not empty, add it
        if title != '':
            plt.title(title)

        plt.savefig(cfg.LRFIND_PLOT_PATH)