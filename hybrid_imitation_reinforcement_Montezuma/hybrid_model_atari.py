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
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda
from keras import optimizers
from keras import initializers

HUBER_DELTA = 0.5

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

    
def clipped_masked_error(args):
        y_true, y_pred, mask = args
        loss = huber_loss(y_true, y_pred, 1)
        loss *= mask  # apply element-wise mask
        return K.sum(loss, axis=-1)

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

    
class Hdqn:
    def __init__(self, gpu):
        self.enable_dueling_network = False
        self.gpu = gpu
        #metrics = [mean_q]
        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        with tf.device('/gpu:'+str(gpu)):
            controller = Sequential()
            controller.add(Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84,84,4)))
            controller.add(Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid'))
            controller.add(Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid'))
            controller.add(Flatten())
            controller.add(Dense(HIDDEN_NODES, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            controller.add(Dense(nb_Action, activation = 'linear', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            controller.compile(loss = 'mse', optimizer = rmsProp)

            controllerTarget = Sequential()
            controllerTarget.add(Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84,84,4)))
            controllerTarget.add(Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid'))
            controllerTarget.add(Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid'))
            controllerTarget.add(Flatten())
            controllerTarget.add(Dense(HIDDEN_NODES, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            controllerTarget.add(Dense(nb_Action, activation = 'linear', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            controllerTarget.compile(loss = 'mse', optimizer = rmsProp)
        
        if not self.enable_dueling_network:
            self.controllerNet = controller
            self.targetControllerNet = controllerTarget
            
            self.controllerNet.reset_states()
            self.targetControllerNet.set_weights(self.controllerNet.get_weights())
        else:
            layer = controller.layers[-2]
            nb_output = controller.output._keras_shape[-1]
            y = Dense(nb_output + 1, activation='linear', kernel_initializer = initializers.random_normal(stddev=0.01))(layer.output)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_output,))(y)

            self.controllerNet = Model(inputs = controller.input, outputs = outputlayer)
            self.controllerNet.compile(optimizer = rmsProp, loss = 'mse')
            self.targetControllerNet = clone_model(self.controllerNet)
            self.targetControllerNet.compile(optimizer = rmsProp, loss = 'mse')
        
    #def saveWeight(self, stepCount):
    #    self.controllerNet.save_weights('controllerNet_' + str(stepCount) + '.h5')
    def saveWeight(self, subgoal):
        self.controllerNet.save_weights(recordFolder+'/policy_subgoal_' + str(subgoal) + '.h5')

    def loadWeight(self, subgoal):
        #path = 'weight/'
        #self.controllerNet.load_weights('standardized_models/policy_subgoal_' + str(subgoal) + '.h5')
        self.controllerNet.load_weights(recordFolder+'/policy_subgoal_' + str(subgoal) + '.h5')
        self.controllerNet.reset_states()
        #self.controllerNet.load_weights('netSubgoal_' + str(subgoal) + '.h5')
        #self.targetControllerNet.load_weights('netSubgoal_' + str(subgoal) + '.h5')
        #self.controllerNet = load_model('netSubgoal_'+str(subgoal)+'.h5', custom_objects = {'huber_loss' :huber_loss})
        #self.targetControllerNet = load_model('netSubgoal_'+str(subgoal)+'.h5', custom_objects = {'huber_loss' :huber_loss})

    def clear_memory(self):
        del self.controllerNet
        del self.targetControllerNet
