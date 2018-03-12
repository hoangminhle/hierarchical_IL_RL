# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

from keras.models import Sequential, Model, load_model, model_from_config
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda
from keras import optimizers
from keras import initializers
import numpy as np
import tensorflow as tf

nb_Action = 8
goalSetSize = 3
np.random.seed(0)

    
class Net:
    def __init__(self):
        #metrics = [mean_q]
        # Refer https://keras.io/getting-started/functional-api-guide/ for creating complex non-sequencial net
        #controller = Sequential()
        #controller.add(Conv2D)
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        #opt = optimizers.Adam(lr=0.0001, clipnorm = 10)
        with tf.device('/gpu:1'):
            self.controller = Sequential()
            self.controller.add(Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84,84,4)))
            self.controller.add(Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid'))
            self.controller.add(Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid'))
            self.controller.add(Flatten())
            self.controller.add(Dense(512, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01)))
            self.controller.add(Dense(nb_Action, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01)))
            self.controller.compile(loss = 'mse', optimizer = rmsProp)

    def sample(self, q_vec, temperature=0.1):
        self._prob_pred = np.log(q_vec) / temperature
        self._dist = np.exp(self._prob_pred)/np.sum(np.exp(self._prob_pred))
        self._choices = range(len(self._prob_pred))
        return np.random.choice(self._choices, p=self._dist)

    def selectMove(self, state, goal):
        #q_array = self.controller.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0)[0]
        q_array = self.controller.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0)
        #return self.sample(q_array)
        #print q_array
        return np.argmax(q_array)

    def loadWeight(self, subgoal):
        #path = 'weight/'
        self.controller.load_weights('trained_models/netSubgoal_' + str(subgoal) + '.h5')
        """
        if subgoal <= 4:
            self.controller.load_weights('netSubgoal_' + str(subgoal) + '.h5')
        else:
            self.controller.load_weights('standardized_models/policy_subgoal_' + str(subgoal) + '.h5')
        """
