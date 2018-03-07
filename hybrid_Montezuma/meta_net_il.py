# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

from hyperparameters import *
"""
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import random
random.seed(SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
"""
from keras.models import Sequential, Model, load_model, model_from_config
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda, Dropout
from keras import optimizers
from keras import initializers

BATCH_SIZE = 32

class MetaNN:
    def __init__(self):
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        with tf.device('/gpu:'+str(GPU)):
            self.meta_controller = Sequential()
            self.meta_controller.add(Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84,84,4)))

            self.meta_controller.add(Dropout(0.5))
            self.meta_controller.add(Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid'))
            self.meta_controller.add(Dropout(0.5))
            self.meta_controller.add(Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid'))
            self.meta_controller.add(Dropout(0.5))

            self.meta_controller.add(Flatten())
            self.meta_controller.add(Dense(HIDDEN_NODES, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            self.meta_controller.add(Dropout(0.5))

            self.meta_controller.add(Dense(nb_Option, activation = 'softmax'))
            self.meta_controller.compile(loss = 'categorical_crossentropy', optimizer = rmsProp)
            self.meta_controller.save_weights("initial_ramdom_weights_metacontroller.h5")

        self.replay_hist = [None] * TRAIN_HIST_SIZE
        self.ind = 0
        self.count = 0
        self.meta_ind = 0

        self.input_shape = (84,84,4)
        self._history = LossHistory()
        self.num_pass = 1

    def reset(self):
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        self.meta_controller.load_weights("initial_ramdom_weights_metacontroller.h5")
        self.meta_controller.compile(loss = 'categorical_crossentropy', optimizer = rmsProp)

    def check_training_clock(self):
        #return (self.ind>=100) and self.count>=10
        return (self.meta_ind>=100)

    def collect(self, processed, expert_a):
        if processed is not None:
            self.replay_hist[self.ind] = (processed.astype(np.float32), expert_a.astype(np.float32))
            self.ind = (self.ind + 1) % TRAIN_HIST_SIZE
            self.count += 1
            self.meta_ind += 1

    def end_collect(self):
        try:
            return self.train()
        except:
            return

    def train(self):
        # if not reached TRAIN_HIST_SIZE yet, then get the number of samples
        self._num_valid = self.ind if self.replay_hist[-1] == None else TRAIN_HIST_SIZE
        try:
            self._samples = range(self._num_valid)
            BATCH_SIZE = len(self._samples)
        except:
            self._samples = range(self._num_valid) + [0] * (BATCH_SIZE - len(range(self._num_valid)))

        # convert replay data to trainable data
        self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
        self._train_x = np.reshape([self._selected_replay_data[i][0] for i in xrange(BATCH_SIZE)],
                                    (BATCH_SIZE,) + self.input_shape)
        self._train_y = np.reshape([self._selected_replay_data[i][1] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,nb_Option))

        self.meta_controller.fit(self._train_x, self._train_y, batch_size = 32, epochs = self.num_pass, callbacks = [self._history])
        self.count = 0 # reset the count clock
        return self._history.losses

    def predict(self, x, batch_size=1):
        """predict on (a batch of) x"""
        return self.meta_controller.predict([np.reshape(x, (1, 84, 84, 4))], batch_size=batch_size, verbose=0)[0]
    def set_pass(self, num_pass):
        self.num_pass = num_pass

    def sample(self, prob_vec, temperature=0.1):
        self._prob_pred = np.log(prob_vec) / temperature
        self._dist = np.exp(self._prob_pred)/np.sum(np.exp(self._prob_pred))
        self._choices = range(len(self._prob_pred))
        return np.random.choice(self._choices, p=self._dist)

import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
