# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

SEED = 1337

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


DEVICE= 'Desktop'
GPU = 0
VERSION = 1

BATCH = 128
TRAIN_FREQ = 4
EXP_MEMORY = 500000
HARD_UPDATE_FREQUENCY = 2000
LEARNING_RATE = 0.0001

# Constant defined here
maxStepsPerEpisode = 499
#maxStepsPerOption = []

goal_to_train = [0,1,2,3]
"""
goalString = str(goal_to_train[0])
for index in range(1,len(goal_to_train)):
	goalString = goalString+str(goal_to_train[index])

recordFolder = "summary_b" +str(BATCH)+ "f"+str(TRAIN_FREQ)+"s90mem"+str(EXP_MEMORY/1000)+"u"+str(HARD_UPDATE_FREQUENCY/1000)+"H"+str(maxStepsPerEpisode+1)+"_goal0123_"+DEVICE+"_v"+str(VERSION)
recordFileName = "hybrid_atari_result_b" +str(BATCH)+ "f"+str(TRAIN_FREQ)+"s90mem"+str(EXP_MEMORY/1000)+"u"+str(HARD_UPDATE_FREQUENCY/1000)+"H"+str(maxStepsPerEpisode+1)+"_goal"+goalString+"_"+DEVICE+"_v"+str(VERSION)
"""
recordFolder = "summary_v"+str(VERSION)
recordFileName = "hybrid_atari_result_v"+str(VERSION)


STOP_TRAINING_THRESHOLD = 0.90
#CLIPNORM = 10

HIDDEN_NODES = 512


#### Things that likely won't change
defaultGamma = 0.99

nb_Action = 8
nb_Option = 4
TRAIN_HIST_SIZE = 10000
EPISODE_LIMIT = 80000
STEPS_LIMIT = 8000000
