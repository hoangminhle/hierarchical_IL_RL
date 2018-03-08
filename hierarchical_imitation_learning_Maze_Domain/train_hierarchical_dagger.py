# Hoang M. Le
# 
# California Institute of Technology
# hmle@caltech.edu
# 
# ===================================================================================================================

from environment_maze import MazeNavigationEnvironment, MazeNavigationStateBuilder
import logging
import math
import os
import sys
import time
import random
import pickle

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
#import keras.backend.tensorflow_backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorboard import TensorboardVisualizer
from datetime import datetime
from os import path
from visualizer import Visualizable
import six

from PIL import Image
from mdp_obstacles import MazeMDP, value_iteration, best_policy

SUMMARY_NAME = 'hda_200WarmStart_1pass_1000maps_randomDoor_run3'
# General parameters
NUM_EPISODES = 1000

SAVE_MODEL_EVERY = 100
#MODEL_NAME = 'dagger_32_32_128_epoch'
#MODEL_NAME = 'go_north_'

HORIZON=100

# Agent parameters
EPSILON       = 1.0      # epsilon-greedy, starting value
EPSILON_END   = 0.1     # epsilon-greedy, ending value
EPSILON_DECAY = 1e-4     # linear decay in epsilon per episode
GAMMA = 0.99             # discount factor

ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
MACRO_ACTIONS = ['N', 'S', 'W', 'E', 'Stay']
OUTPUT_MACRO_DIM = 5

# NN parameters
TRAIN_HIST_SIZE = 100000
BATCH_SIZE = 32

CHANNELS = 3
INPUT_HEIGHT = 16 # HEIGHT
INPUT_WIDTH = 16 # WIDTH
OUTPUT_DIM = 4


TRAIN_FREQUENCY = 100
LEARNING_RATE = 0.0005

# Logging
LOG_LEVEL = logging.DEBUG

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
logger.handlers = []
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
logger.addHandler(logging.StreamHandler(sys.stdout))

import keras
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Input
from keras.models import Model

class MetaNN:
	def __init__(self, input_shape):
		with tf.device('/gpu:1'):
			inputs = Input(shape = input_shape)

			x = Conv2D(32, kernel_size = 3, strides = (1,1), padding = 'same', data_format = 'channels_last', activation = 'relu')(inputs)
			x = Conv2D(32, kernel_size = 3, strides = (1,1),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			x = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(x)
			x = Conv2D(64,(3,3),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			conv_out = Flatten()(x)
			y = Dense(256, activation = 'relu')(conv_out)
			y = Dropout(0.5)(y)
			option_outputs = Dense(OUTPUT_MACRO_DIM, activation = 'softmax', name = 'action_outputs')(y)

			self.model = Model(inputs = inputs, outputs = option_outputs)
			self.model.compile(loss = 'categorical_crossentropy', 
								optimizer = Adam(lr = LEARNING_RATE))

		# Print out model dimensions
		logger.warning('Model input dim: ' + str(self.model.layers[0].input_shape))
		for l in self.model.layers:
			logger.warning('Output dim: ' + str(l.output_shape))
		# store history of length TRAIN_HIST_SIZE
		#time.sleep(30)
		self.replay_hist = [None] * TRAIN_HIST_SIZE
		self.ind = 0
		self.count = 0

		self.input_shape = input_shape
		self._history = LossHistory()
		self.num_pass = 1

	def check_training_clock(self):
		#return (self.ind>=100) and self.count>=10
		return (self.ind>=100)

	def collect(self, processed, expert_a):
		if processed is not None:
			self.replay_hist[self.ind] = (processed.astype(np.float32), expert_a.astype(np.float32))
			self.ind = (self.ind + 1) % TRAIN_HIST_SIZE
			self.count += 1

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
		self._train_y = np.reshape([self._selected_replay_data[i][1] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,5))

		self.model.fit(self._train_x, self._train_y, batch_size = 32, epochs = self.num_pass, callbacks = [self._history])
		self.count = 0 # reset the count clock
		return self._history.losses

	def predict(self, x, batch_size=1):
		"""predict on (a batch of) x"""
		return self.model.predict(x, batch_size=batch_size, verbose=0)
	def set_pass(self, num_pass):
		self.num_pass = num_pass


class NN:
	def __init__(self, input_shape):
		with tf.device('/gpu:1'):
			inputs = Input(shape = input_shape)

			x = Conv2D(32, kernel_size = 3, strides = (1,1), padding = 'same', data_format = 'channels_last', activation = 'relu')(inputs)
			x = Conv2D(32, kernel_size = 3, strides = (1,1),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			x = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(x)
			x = Conv2D(64,(3,3),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			conv_out = Flatten()(x)
			y = Dense(256, activation = 'relu')(conv_out)
			y = Dropout(0.5)(y)
			action_outputs = Dense(OUTPUT_DIM, activation = 'softmax', name = 'action_outputs')(y)

			self.model = Model(inputs = inputs, outputs = action_outputs)
			self.model.compile(loss = 'kullback_leibler_divergence', 
								optimizer = Adam(lr = LEARNING_RATE))

			inputs_termination = Input(shape = input_shape)
			z = Conv2D(8, kernel_size = 5, strides = (1,1), data_format = 'channels_last', activation = 'relu')(inputs_termination)
			z = Dropout(0.5)(z)
			conv_out_termination = Flatten()(z)
			
			t = Dense(64, activation = 'relu')(conv_out_termination)
			t = Dropout(0.5)(t)
			goal_output = Dense(1, activation = 'sigmoid', name = 'goal_output')(t)

			self.model_terminate = Model(inputs = inputs_termination, outputs = goal_output)
			self.model_terminate.compile(loss = 'binary_crossentropy', 
								optimizer = Adam(lr = LEARNING_RATE))

		# Print out model dimensions
		logger.warning('Model input dim: ' + str(self.model.layers[0].input_shape))
		for l in self.model.layers:
			logger.warning('Output dim: ' + str(l.output_shape))
		# store history of length TRAIN_HIST_SIZE
		#time.sleep(30)
		self.replay_hist = [None] * TRAIN_HIST_SIZE
		self.ind = 0
		self.count = 0

		self.input_shape = input_shape
		self._history = LossHistory()
		self.num_pass = 1

	def check_training_clock(self):
		#return (self.ind>=100) and self.count>=10
		return (self.ind>=100)
	def collect(self, processed, expert_a, goal, initiation_point):
		if processed is not None:
			self.replay_hist[self.ind] = (processed.astype(np.float32), expert_a.astype(np.float32), goal, initiation_point.astype(np.float32))
			self.ind = (self.ind + 1) % TRAIN_HIST_SIZE
			self.count += 1

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
		self._train_y = np.reshape([self._selected_replay_data[i][1] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,4))

		self._train_x_difference = np.reshape([self._selected_replay_data[i][0] - self._selected_replay_data[i][3] for i in xrange(BATCH_SIZE)],
									(BATCH_SIZE,) + self.input_shape)
		self._train_g = np.reshape([self._selected_replay_data[i][2] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,1))

		#self.model.fit(self._train_x, [self._train_y, self._train_g], batch_size = 32, epochs = self.num_pass, callbacks = [self._history])
		self.model.fit(self._train_x, self._train_y, batch_size = 32, epochs = self.num_pass, callbacks = [self._history])
		self.model_terminate.fit(self._train_x_difference, self._train_g, batch_size = 32, epochs = self.num_pass)
		self.count = 0 # reset the count clock
		return self._history.losses

	def predict(self, x, batch_size=1):
		"""predict on (a batch of) x"""
		return self.model.predict(x, batch_size=batch_size, verbose=0)
	def predict_termination(self, x, batch_size=1):
		"""predict on (a batch of) x"""
		return self.model_terminate.predict(x, batch_size=batch_size, verbose=0)

	def set_pass(self, num_pass):
		self.num_pass = num_pass


# Agent logic
class Agent(Visualizable):
	def __init__(self, mode, env, direction):
		self.input_shape = (17, 17, CHANNELS)
		self.model = [NN(self.input_shape) for index in range(5)]
		self.metacontroller = MetaNN(self.input_shape)

		self.direction = direction

		#self.expert = expert
		self.mode = mode
		if mode == 'train':
			logdir = path.join('summary/hierarchical_dagger/', SUMMARY_NAME) ## subject to change
			self._visualizer = TensorboardVisualizer()
			self._visualizer.initialize(logdir,None)
		
		self.total_steps = 0

		self._stats_loss = []
		#self._stats_rewards = []
		self._stats_meta_loss = []
		self._stats_val_rewards = 0
		self._success = 0
		self._stats_success = []
		self._stats_performance = [] 

		self.experience = [None] * TRAIN_HIST_SIZE ## Store transitions here, for all options
		self.meta_experience = [None] * TRAIN_HIST_SIZE
		self.ind = 0 ## index keeper for all options
		self.meta_ind = 0

		self._macro_a = -1 ## -1 means request new macro action?
		self._complete_episode = 0
		self._incomplete_episode = 0
		self.teach_episode = 0

		self.warmstart = True
		if self.warmstart:
			self.metacontroller.set_pass(1)
			for index in range(5):
				self.model[index].set_pass(1)


	def turn_off_warmstart(self):
		self.warmstart = False
		for index in range(5):
			self.model[index].set_pass(1)
		self.metacontroller.set_pass(1)

	def collect_meta_experience(self, agent_loc, expert_macro_action, agent_macro_action):
		self.meta_experience[self.meta_ind] = (agent_loc, expert_macro_action, agent_macro_action)
		self.meta_ind = (self.meta_ind + 1) % TRAIN_HIST_SIZE

	def collect_experience(self, agent_loc, prev_state, agent_action, reward, next_state, expert_advice):
		self.experience[self.ind] = (agent_loc, prev_state.astype(np.float32), agent_action, reward, next_state.astype(np.float32), expert_advice.astype(np.float32))
		self.ind = (self.ind + 1) % TRAIN_HIST_SIZE

	def save_experience(self, fileName):
		with open(fileName, 'wb') as f:
			pickle.dump(self.experience, f, protocol=pickle.HIGHEST_PROTOCOL)

	def save_success_record(self, fileName):
		with open(fileName, 'wb') as f:
			pickle.dump(self._stats_performance, f, protocol=pickle.HIGHEST_PROTOCOL)

	def change_expert(self, list_of_dictionary):
		self.expert = list_of_dictionary ## should be the list of 5 experts for 5 options, goal option is the same as the whole q-table 
		#self.expert = tables[index].copy()

	def sample(self, prob_vec, temperature=0.1):
		self._prob_pred = np.log(prob_vec) / temperature
		self._dist = np.exp(self._prob_pred)/np.sum(np.exp(self._prob_pred))
		self._choices = range(len(self._prob_pred))
		return np.random.choice(self._choices, p=self._dist)

	def get_expert_policy(self, agent_host):
		reward_description = []
		terminal_description = []
		for x in range(17):
		    row_reward = []
		    for y in range(17):
		        if agent_host._world[x, y] == 'o':
		            row_reward.append(-0.01)
		        elif agent_host._world[x, y] == 'a' or agent_host._world[x, y] == 'w':
		            row_reward.append(-0.01)            
		        elif agent_host._world[x, y] == 'x':
		            row_reward.append(-1)
		            terminal_description.append((x,y))
		        elif agent_host._world[x, y] == 'g':
		            row_reward.append(1)
		            terminal_description.append((x,y))
		    reward_description.append(row_reward)
		maze = MazeMDP(reward_description, terminal_description, init = agent_host.agent_loc, gamma = 0.99)
		self.expert_value = value_iteration(maze)
		self.policy = best_policy(maze, self.expert_value)
		self.expert= {}
		
		for row in range(17):
		    for col in range(17):
		        if self.policy[(col, row)] == (0,-1):
		            self.expert[(col,row)] = 0 #N
		        elif self.policy[(col, row)] == (0,1):
		            self.expert[(col,row)] = 1 #S
		        elif self.policy[(col, row)] == (-1,0):
		            self.expert[(col,row)] = 2 #W
		        elif self.policy[(col, row)] == (1,0):
		            self.expert[(col,row)] = 3 #E

	def get_expert_trajectory(self, agent_host, starting_loc, goal_loc):
		self.get_expert_policy(agent_host)

		self.sa_trajectory = []
		print " starting location: ", starting_loc
		state = [starting_loc[0], starting_loc[1]]
		while not (state[0] == goal_loc[0] and state[1] == goal_loc[1]):
			action = self.expert[(state[0], state[1])]
			action_direction = self.policy[(state[0], state[1])]
			self.sa_trajectory.append(( (state[0], state[1]), action))

			old_state_x = state[0]
			old_state_y = state[1]
			state = [old_state_x + action_direction[0], old_state_y + action_direction[1]]

		self.room_trajectory = []
		self.expert_macro_feedback = {}
		for item in self.sa_trajectory:
			i,j = item[0]
			if i % 4 == 0 or j %4 == 0:
				room= (-1,-1) # hall way
			else:
				room = (i /4, j /4)
			self.room_trajectory.append(room)
		
		## attribute the hallway back to the preceding room, make sure it is correct
		change_point = []
		change_point.append(0) ## the first location
		for index in range(len(self.room_trajectory)):
			if self.room_trajectory[index] == (-1,-1):
				assert index >0
				self.room_trajectory[index] = self.room_trajectory[index-1] 
				change_point.append(index+1) # the next index should be the terminal point of an option, and should be an intermediary location
		change_point.append(len(self.room_trajectory)) ## the last location

		print "length of room trajectory ", len(self.room_trajectory)
		
		### now figure out the meta-actions
		for index in range(len(change_point)-1):
			begin_index = change_point[index]
			end_index = change_point[index+1]
			if end_index < len(self.room_trajectory):
				room_shift = (self.room_trajectory[end_index][0] - self.room_trajectory[begin_index][0], self.room_trajectory[end_index][1] - self.room_trajectory[begin_index][1])
				#print "room shift value is ", room_shift
				assert room_shift != (0,0)
				if room_shift == (0,-1):
					option = 0
				elif room_shift == (0,1):
					option = 1
				elif room_shift == (-1,0):
					option = 2
				elif room_shift == (1,0):
					option = 3
				else:
					option = 4
			else:
				option = 4


			for i in range(begin_index, end_index):
				state = self.sa_trajectory[i][0]
				self.expert_macro_feedback[state] = option

		self.expert_micro_feedback = {}
		for index in range(len(self.room_trajectory)):
			state = self.sa_trajectory[index][0]
			action = self.sa_trajectory[index][1]
			if index in change_point and index >0:
				self.expert_micro_feedback[state] = (action, 1)
			else:
				self.expert_micro_feedback[state] = (action, 0)

	def choose_macro_action(self, world_state):
		state = np.reshape(world_state, (1,)+self.input_shape)
		self._macro_action_pred = self.metacontroller.predict(state)[0]
		self._macro_a = self.sample(self._macro_action_pred)

	def get_expert_feedback(self, agent_host, macro_action, expert_policy, decision_point):
		agent_loc = agent_host.agent_loc
		#print "agent loc:", agent_loc
		#print "decision point:", decision_point
		if agent_host._world[agent_loc[0], agent_loc[1]] == 'x':
			possible_actions = [(0,-1), (0,1), (-1,0), (1,0)]
			best_direction = (0,0)
			max_value = -1.0
			for action in range(4):
				direction = possible_actions[action]
				state = (agent_loc[0]+direction[0], agent_loc[1]+direction[1])
				if state[0]>=0 and state[0]<=16 and state[1]>=0 and state[1]<=16 and agent_host._world[state[0], state[1]] == 'w':
					if self.expert_value[state] > max_value:
						max_value = self.expert_value[state]
						best_direction = direction
			expert_a = possible_actions.index(best_direction)
			expert_t = 0

		else:
			expert_a = expert_policy[agent_loc]
			if macro_action == 0:
				if agent_loc[1] % 4 == 3 and agent_loc[1] < 15 and agent_loc[1] < decision_point[1] and (agent_loc[0] / 4) == (decision_point[0] /4):
					expert_t = 1
				else:
					expert_t =  0
			elif macro_action == 1:
				if agent_loc[1] % 4 == 1 and agent_loc[1] > 1 and agent_loc[1] > decision_point[1] and (agent_loc[0] / 4) == (decision_point[0] /4):
					expert_t= 1
				else:
					expert_t= 0
			elif macro_action == 2:
				if agent_loc[0] % 4 == 3 and agent_loc[0] < 15 and agent_loc[0] < decision_point[0] and (agent_loc[1] / 4) == (decision_point[1] /4):
					expert_t= 1
				else:
					expert_t= 0
			elif macro_action == 3:
				if agent_loc[0] % 4 == 1 and agent_loc[0] > 1 and agent_loc[0] > decision_point[0] and (agent_loc[1] / 4) == (decision_point[1] /4):
					expert_t= 1
				else:
					expert_t= 0
			elif macro_action == 4:
				expert_t= 0

			if expert_t == 0 and macro_action != self.identify_optimal_macro_action(agent_host, agent_loc):
				print "not optimal meta action chosen"
				expert_a = 99
				expert_t = 99
		#else:

		return expert_a, expert_t


	def identify_optimal_macro_action(self, agent_host, starting_loc):
		if agent_host._world[starting_loc[0], starting_loc[1]] == 'x':
			print "recursive call to determine best meta action"
			possible_actions = [(0,-1), (0,1), (-1,0), (1,0)]
			best_direction = (0,0)
			max_value = -1.0
			for action in range(4):
				direction = possible_actions[action]
				state = (agent_loc[0]+direction[0], agent_loc[1]+direction[1])
				if state[0]>=0 and state[0]<=16 and state[1]>=0 and state[1]<=16 and agent_host._world[state[0], state[1]] == 'w':
					if self.expert_value[state] > max_value:
						max_value = self.expert_value[state]
						best_direction = direction
			assert best_direction != (0,0)
			new_loc = (starting_loc[0] + best_direction[0], starting_loc[1] + best_direction[1])
			return self.identify_optimal_macro_action(self, agent_host, new_loc)
		else:
			state = [starting_loc[0], starting_loc[1]]
			meta_action_terminate = False

			if starting_loc[0] % 4==0 or starting_loc[1]%4 == 0:
				starting_room = (-1,-1)
			else:
				starting_room = (starting_loc[0] /4, starting_loc[1] /4)
			
			#print "starting room:", starting_room
			goal_room = (agent_host.goal_loc[0] / 4, agent_host.goal_loc[1]/4)
			if starting_room == goal_room:
				expert_macro_a = 4
			else:
				if starting_room == (-1,-1):
					room_shift = self.policy[(state[0], state[1])]
					#print "action direction from hallway:", room_shift
					#time.sleep(5)
				else:
					#print "here"
					while not meta_action_terminate:
						action = self.expert[(state[0], state[1])]
						action_direction = self.policy[(state[0], state[1])]

						old_state_x = state[0]
						old_state_y = state[1]
						state = [old_state_x + action_direction[0], old_state_y + action_direction[1]]

						## check if meta action should terminate
						if state[0] % 4==0 or state[1]%4 == 0:
							current_room = (-1,-1)
						else:
							current_room = (state[0] /4, state[1] /4)
						if current_room != starting_room and current_room != (-1,-1):
							meta_action_terminate = True
							#print "current room", current_room

					room_shift = (current_room[0] - starting_room[0], current_room[1] - starting_room[1])
				assert room_shift != (0,0)
				#print "room shift:", room_shift
				if room_shift == (0,-1):
					expert_macro_a = 0
				elif room_shift == (0,1):
					expert_macro_a = 1
				elif room_shift == (-1,0):
					expert_macro_a = 2
				elif room_shift == (1,0):
					expert_macro_a = 3
				else:
					print "something is wrong when identifying optimal macro action"
			return expert_macro_a
	
	def learner_choose_option(self, agent_host):
		world_state = agent_host.state
		state = np.reshape(world_state, (1,)+self.input_shape)
		self._macro_action_pred = self.metacontroller.predict(state)[0]
		self._macro_a = self.sample(self._macro_action_pred)
		### double check to prevent loops
		#print "train- originally wanted to pick option ", self._macro_a, " at ", agent_host.agent_loc
		tau = self.model[self._macro_a].predict_termination(state - self.initiation_point)[0]

		if tau>=0.9:
			print "train - had to pick another option to prevent loops"
			self._macro_action_pred[self._macro_a] = 0.0 # disable this option
			resum = sum(self._macro_action_pred) * 1.0
			self._macro_action_pred = self._macro_action_pred / resum
			self._macro_a = self.sample(self._macro_action_pred)
			#print "train - forced to change the pick to ", self._macro_a, " at ", agent_host.agent_loc
		#else:
		#	print "train - pick not modified"

	def check_goal_achieve(self, agent_host,option_index,last_location, decision_point):
		agent_loc = last_location
		macro_action = option_index

		if macro_action == 0:
			if agent_loc[1] % 4 == 3 and agent_loc[1] < 15 and agent_loc[1] < decision_point[1] and (agent_loc[0] / 4) == (decision_point[0] /4):
				expert_t = 1
			else:
				expert_t =  0
		elif macro_action == 1:
			if agent_loc[1] % 4 == 1 and agent_loc[1] > 1 and agent_loc[1] > decision_point[1] and (agent_loc[0] / 4) == (decision_point[0] /4):
				expert_t= 1
			else:
				expert_t= 0
		elif macro_action == 2:
			if agent_loc[0] % 4 == 3 and agent_loc[0] < 15 and agent_loc[0] < decision_point[0] and (agent_loc[1] / 4) == (decision_point[1] /4):
				expert_t= 1
			else:
				expert_t= 0
		elif macro_action == 3:
			if agent_loc[0] % 4 == 1 and agent_loc[0] > 1 and agent_loc[0] > decision_point[0] and (agent_loc[1] / 4) == (decision_point[1] /4):
				expert_t= 1
			else:
				expert_t= 0
		elif macro_action == 4:
			expert_t= 0

		return expert_t

	def run_train(self, agent_host):
		## calculate the expert
		self.get_expert_policy(agent_host)
		self.train_flag = False
		self.rerun = False
		meta_labels = []
		meta_controller_call_history = []

		"""run the agent on the world"""
		self.expert_label = 0
		self._total_reward = 0
		self._step = 0

		self.trajectory = []
		self.terminate = False

		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1)
		
		self._world_state = agent_host.state
		self._state = np.reshape(self._world_state, (1,)+self.input_shape)

		# create initiation point in order to determine the correct termination condition
		self.decision_point = agent_host.agent_loc
		self.initiation_point = np.reshape(agent_host.state, (1,)+self.input_shape) # first initiation point

		self.learner_choose_option(agent_host)
		#self.expert_macro_a, self.bad_set = self.identify_optimal_macro_action(agent_host, agent_host.agent_loc, self._final_goal_loc, self._final_goal_room, self.expert[4])
		expert_macro_a = self.identify_optimal_macro_action(agent_host, agent_host.agent_loc)

		## Note: add all meta-labels
		self.expert_macro_a = np.zeros((1,len(MACRO_ACTIONS)))
		self.expert_macro_a[0, expert_macro_a] = 1.0
		if meta_controller_call_history.count(agent_host.agent_loc) >=1:
			print "Already request meta label here at least once. We're not adding more"
		else:
			meta_controller_call_history.append(agent_host.agent_loc)
			meta_labels.append((self._state, self.expert_macro_a)) # append, but do not collect yet

		if self.warmstart or self._macro_a != expert_macro_a:
			if self.warmstart:
				print "warmstart phase - adding meta data"
				self.train_flag = True
			elif self._macro_a != expert_macro_a and self.rerun_count <= 10:
				print "option selected not optimal, hold the training flag"
			elif self._macro_a != expert_macro_a and self.rerun_count > 10:
				print "option selected not optimal, turn on training flag"
				self.train_flag = True
			#self.metacontroller.collect(self._state, self.expert_macro_a) 
			#self.meta_ind += 1
		if self.warmstart:
			self._macro_a = expert_macro_a

		#print
		#print "train - option chosen:", self._macro_a
		#time.sleep(2)
		
		option_segment = []
		#time.sleep(1)

		while (not self.terminate) and (not agent_host.done) and self._step< HORIZON:
			self._world_state = agent_host.state
			self._state = np.reshape(self._world_state, (1,)+self.input_shape)

			expert_a, expert_t = self.get_expert_feedback(agent_host, self._macro_a, self.expert, self.decision_point)
			self.expert_a = np.zeros((1,len(ACTIONS)))
			if expert_a != 99: #only give expert feedback when the macro action is optimal
				self.expert_a[0,expert_a] = 1.

			agent_loc = agent_host.agent_loc

			if self.warmstart:
				self._a = expert_a
				self._tau = expert_t
			else:
				self._action_prob = self.model[self._macro_a].predict(self._state)[0]
				self._tau = self.model[self._macro_a].predict_termination(self._state - self.initiation_point)[0]
				self._a = self.sample(self._action_prob)

			experience = (agent_loc, self._state, self._macro_a, self.expert_macro_a, self._a, self._tau, self.expert_a, expert_t, self.initiation_point, self.decision_point)
			option_segment.append(experience)
			
			if self._tau < 0.9:
				agent_host._update(ACTIONS[self._a]) # take action
				reward = agent_host.reward
				# check to see if we're in a loop
				if (( len(option_segment)>=5)
					and (option_segment[-1][0] == option_segment[-3][0]) and (option_segment[-1][4] == option_segment[-3][4])
					and (option_segment[-1][0] == option_segment[-5][0]) and (option_segment[-1][4] == option_segment[-5][4]) ): # same spot and same action taken
					print
					print "looks like getting in a loop, send a break signal"
					self.train_flag = True
					self.terminate = True
				#else:
					#self.collect_experience(agent_host.agent_loc, prev_state, self._a, reward, agent_host.state, expert_a) # record the transition

				if agent_host._rendering:
					agent_host.render()
					time.sleep(0.1)
				self._step += 1
				self._total_reward += reward
			else:
				#print
				#print "learner chooses to terminate at", agent_loc
				# close the book on the current option
				self.trajectory.append(option_segment)
				option_segment = []
				#time.sleep(1)

				# create initiation point in order to determine the correct termination condition
				self.decision_point = agent_host.agent_loc
				self.initiation_point = np.reshape(agent_host.state, (1,)+self.input_shape) # first initiation point

				self.learner_choose_option(agent_host)
				#self.expert_macro_a, self.bad_set = self.identify_optimal_macro_action(agent_host, agent_host.agent_loc, self._final_goal_loc, self._final_goal_room, self.expert[4])
				expert_macro_a = self.identify_optimal_macro_action(agent_host, agent_host.agent_loc)

				### Note: always add meta labels
				self.expert_macro_a = np.zeros((1,len(MACRO_ACTIONS)))
				self.expert_macro_a[0, expert_macro_a] = 1.0
				if meta_controller_call_history.count(agent_host.agent_loc) >=1:
					print "Already request meta label here at least once. We're not adding more"
				else:
					meta_controller_call_history.append(agent_host.agent_loc)
					meta_labels.append((self._state, self.expert_macro_a)) # append, but do not collect yet

				if self.warmstart or self._macro_a != expert_macro_a:
					if self.warmstart:
						print "warmstart phase - adding meta data"
						self.train_flag = True
					elif self._macro_a != expert_macro_a and self.rerun_count <= 50:
						print "option selected not optimal, hold the training flag"
					elif self._macro_a != expert_macro_a and self.rerun_count > 50:
						print "option selected not optimal, turn on training flag"
						self.train_flag = True
						#self.train_flag = True
					#self.metacontroller.collect(self._state, self.expert_macro_a)
					#self.meta_ind += 1
				if self.warmstart:
					self._macro_a = expert_macro_a

				#print
				#print "next chosen macro action is then:" ,self._macro_a
				
				#print "teach value is now", self._teach
		
		## if terminate, add the last option segment into the trajectory:
		## appent the last reward and experience:
		if agent_host.done:
			#final_experience = (agent_host.agent_loc, agent_host.state, self._macro_a, self.expert_macro_a, [], -100, 1.0, np.ones((1,len(ACTIONS))) /4.0, 1.0)
			final_experience = (agent_host.agent_loc, self._state, self._macro_a, self.expert_macro_a, self._a, 0.0, self.expert_a, 0.0, self.initiation_point, self.decision_point) ## intentionally avoid 1 for tau, for goal
			option_segment.append(final_experience)
		self.trajectory.append(option_segment)

		final_reward = agent_host.reward
		if final_reward < 1:
			self.train_flag = True
		#final_reward = agent_host.reward

		## remains to process final reward and other stuff
		#### expert replay the trajectory to decide where to actually incorporate feedback:
		print "number of segments:", len(self.trajectory)
		print "actual number of meta labels to be collected:", len(meta_controller_call_history)
		#time.sleep(0.5)
		if self.warmstart:
			## collect data for meta controller
			for item in meta_labels:
				self.metacontroller.collect(item[0], item[1])
				self.meta_ind += 1
			## collect data for each controller
			for index in range(len(self.trajectory)):
				option_segment = self.trajectory[index]
				self.ind = self.ind + len(option_segment)
				#self.meta_ind = self.meta_ind + 1
				for item in option_segment:
					#print "experience tuple (loc, option, expert_action, self_t, t):", item[0], item[2], item[7], item[6], item[8]
					#print item[6]
					if sum(item[6][0]) < 1:
						print "option in badset - check"
					else:
						self.model[item[2]].collect(item[1], item[6], item[7],item[8])

					#self.meta_ind += 1
					#self.metacontroller.collect(item[1], item[3])
		else:
			if self.train_flag:
				for item in meta_labels:
					self.metacontroller.collect(item[0], item[1])
					self.meta_ind += 1				

				for index in range(len(self.trajectory)):
					option_segment = self.trajectory[index]
					#check termination point:
					option_index = option_segment[-1][2]
					assert option_index == option_segment[0][2] ## make sure the options executed in the segment is consistent
					last_location = option_segment[-1][0] # agent location at the last point before closing the option
					decision_point = option_segment[-1][9]
					assert decision_point == option_segment[0][9] ## make sure the initiation point stays consistent throughout the segment
					#print
					#print "all locations for option ", option_index
					#print "the last location of option ", option_index, " is ", last_location
					# check if goal achieve?
					if option_index == np.argmax(option_segment[0][3][0]): #if the option executed is the same as the optimal option at the beginning of the segment, then it is good
						print "Good option ", option_index, " being executed"
						goal_achieved = self.check_goal_achieve(agent_host,option_index,last_location, decision_point)
						#print "goal achieved? ", goal_achieved
						if not goal_achieved:
							print "Goal not achieved for meta action:", option_index
							label_count = 0
							#self.ind = self.ind + len(option_segment)
							for item in option_segment:
								#print "experience tuple (loc, self option, expert option, expert_action, self_t, t):", item[0], item[2], item[3], item[7], item[6], item[8]
								#if item[2] != np.argmax(item[3][0]):
								if sum(item[6][0])<1:
									print "expert advice for this is:",item[6][0], "so not collecting"
								else:
									self.model[item[2]].collect(item[1], item[6], item[7], item[8])
									self.ind += 1
									label_count +=1
							print "Added ", label_count, " labels for training option ", option_index
							if label_count >0:
								self.ind = self.ind - 1 # take one back to avoid double counting the meta actions
					else:
						print "bad option ", option_index, " attempted. Will ignore"
		if agent_host._rendering:
			time.sleep(0.1)
		
		if self.train_flag and self.metacontroller.check_training_clock():
			print
			print "training metacontroller"
			meta_loss = self.metacontroller.train()
			self._stats_meta_loss.append(sum(meta_loss)/len(meta_loss))
		if self.train_flag:
			for index in range(5):
				#if self.model[index].check_training_clock():
				if self.model[index].check_training_clock():
					print
					print "training sub-goal", index
					subgoal_loss = self.model[index].train()
					self._stats_loss.append(sum(subgoal_loss)/len(subgoal_loss))
		if not self.train_flag:
			print
			print "Good Execution - Let's skip this episode"
			self.rerun = True
			self.rerun_count += 1
		else:
			self.rerun_count = 0
		"""
		self._stats_success.append(self._success)
		self._success = 0
		"""

		#return self._total_reward
		return final_reward


	def choose_option_test(self, agent_host):
		world_state = agent_host.state
		state = np.reshape(world_state, (1,)+self.input_shape)
		self._macro_action_pred = self.metacontroller.predict(state)[0]
		self._macro_a = self.sample(self._macro_action_pred)
		#print "test - originally wanted to pick option ", self._macro_a, " at ", agent_host.agent_loc
		### double check to prevent loops
		#tau = self.model[self._macro_a].predict(state)[1][0]
		tau = self.model[self._macro_a].predict_termination(state-self.initiation_point)[0]
		if tau>=0.9:
			print "test - had to pick another option to prevent loops"
			self._macro_action_pred[self._macro_a] = 0.0 # disable this option
			resum = sum(self._macro_action_pred) * 1.0
			self._macro_action_pred = self._macro_action_pred / resum
			self._macro_a = self.sample(self._macro_action_pred)
			#print "test - forced to change the pick to ", self._macro_a, " at ", agent_host.agent_loc
		#else:
		#	print "test - pick not modified"


	def run_test(self, agent_host):
		self._total_reward = 0
		self._step = 0
		self._option_step = 0

		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1)
		self._world_state = agent_host.state
		self._state = np.reshape(self._world_state, (1,)+self.input_shape)

		self.initiation_point = np.reshape(agent_host.state, (1,)+self.input_shape) # first initiation point

		self.choose_option_test(agent_host)
		self._option_step += 1
		print
		#print "test - option chosen:", self._macro_a
		while (not agent_host.done) and self._step< HORIZON:
			self._world_state = agent_host.state
			self._state = np.reshape(self._world_state, (1,)+self.input_shape)

			#self._action_prob = self.model[self._macro_a].predict(self._state)[0][0]
			self._action_prob = self.model[self._macro_a].predict(self._state)[0]
			
			self._tau = self.model[self._macro_a].predict_termination(self._state - self.initiation_point)[0]
			#self._tau = self.model[self._macro_a].predict(self._state)[1][0]
			self._a = self.sample(self._action_prob)
			if self._tau < 0.9:
				agent_host._update(ACTIONS[self._a]) # take action
				reward = agent_host.reward
				if agent_host._rendering:
					agent_host.render()
					time.sleep(0.1)
				self._step += 1
				self._total_reward += reward
			else:
				#print
				#print "test - learner chooses to terminate at:", agent_host.agent_loc
				self.initiation_point = np.reshape(agent_host.state, (1,)+self.input_shape) # first initiation point
				self.choose_option_test(agent_host)
				#print "test - next chosen macro action is then:" ,self._macro_a
				self._option_step += 1
		print
		print "total number of options called:", self._option_step
		if agent_host.reward > 0:
			self._success = 1
			print "Success"
		elif agent_host.reward == -1.0: 
			print "Bumped into lava"
		else:
			print "Episode not finished"
		
		self._stats_success.append(self._success)
		self._stats_performance.append((self._success, self.ind))
		self._success = 0

		return self._total_reward

	def inject_summaries(self, idx):
		if self.mode == 'train':
			
			if len(self._stats_loss) > 0:
				self.visualize(idx, "episode loss",
							   np.asscalar(np.mean(self._stats_loss)))
			
			if len(self._stats_meta_loss) > 0:
				self.visualize(idx, "episode meta loss",
							   np.asscalar(np.mean(self._stats_meta_loss)))
			
			#self.visualize(idx, "number of incomplete episode", self._incomplete_episode)
			#self.visualize(idx, "number of complete episode", self._complete_episode)
			self.visualize(idx, "expert labels for meta controller", self.meta_ind)
			self.visualize(idx, "expert labels for primitive controller", self.ind)


			## Showing the success rate for the trailing 20 training episode
			#if not self.warmstart:
			if len(self._stats_success) >100:
				self.visualize(idx, "episode success indicator", np.asscalar(np.mean(self._stats_success[-100:])))
				self.visualize(self.teach_episode, "adjusted episode success indicator", np.asscalar(np.mean(self._stats_success[-100:])))
				#self.visualize(self.ind+self.meta_ind, "learning curve",np.asscalar(np.mean(self._stats_success[-100:])))
				self.visualize(self.ind + self.meta_ind, "learning curve",np.asscalar(np.mean(self._stats_success[-100:])))
			else:
				self.visualize(idx, "episode success indicator", np.asscalar(np.mean(self._stats_success)))
				#self.visualize(self.ind+self.meta_ind, "learning curve",np.asscalar(np.mean(self._stats_success)))
				self.visualize(self.ind, "learning curve",np.asscalar(np.mean(self._stats_success)))

				# Reset
			self._stats_loss = []
			self._stats_meta_loss = []
	def load_model(self, fileName):
		self.model.model.load_weights(fileName)

def main(macro_action, train_or_test, environment, validation):
	direction = macro_action
	mode = train_or_test

	PRIMITIVE_MODEL_NAME = 'subgoal_HDA_'
	METACONTROLLER_NAME = 'metacontroller_HDA_'

	agent_host = MazeNavigationEnvironment(MazeNavigationStateBuilder(gray = False),
									rendering = False, randomized_door=True, stochastic_dynamic = False, map_id=999, setting = environment)
	
	agent = Agent(mode, environment, direction)

	map_ids_to_use = range(1000,2000)

	#for i in range(1, 20000):
	for i in six.moves.range(1,NUM_EPISODES+1):
		print 
		print "--------------------------------------------------------------------"
		logger.info("\nMission %d of %d:" % ( i, NUM_EPISODES ))

		new_map_id = np.random.choice(range(1000))
		print "Launching map", new_map_id

		agent_host.change_map_and_reset(new_map_id)
		
		world_state = agent_host.state

		# -- run the agent in the world -- #
		if i >200:
			agent.turn_off_warmstart()

		if agent.warmstart:
			final_reward = agent.run_train(agent_host)
		else:
			final_reward = agent.run_train(agent_host)
			print "final reward for this training round:", final_reward
			while agent.rerun:
				print
				print "********rerun since didnt learn anything last run****"
				#agent.train_buffer += 1
				print
				print "re-attempt number:", agent.rerun_count
				new_map_id = np.random.choice(range(1000))
				agent_host.change_map_and_reset(new_map_id)

				final_reward = agent.run_train(agent_host)
				print "final reward for this training round:", final_reward
		
		## run a proper validation
		print "------begin test"
		new_map_id = np.random.choice(map_ids_to_use)
		agent_host.change_map_and_reset(new_map_id)
		
		cumulative_reward = agent.run_test(agent_host)

		agent.inject_summaries(i)


		logger.info("Cumulative reward: " + str(cumulative_reward))
		
		if i % SAVE_MODEL_EVERY == 0: # throw away the first 1000 episodes?
			print "to save"
			agent.save_success_record('summary/success_record/result_'+SUMMARY_NAME+'_'+str(int(i/SAVE_MODEL_EVERY))+'.pkl')
			#agent.model.model.save_weights(MODEL_NAME, overwrite = True)


	logger.warning("Done.")


if __name__ == '__main__':
	arg_parser = ArgumentParser('Hierarchical Dagger experiment')
	arg_parser.add_argument('-d', '--direction', type=str, choices=['north', 'south', 'west', 'east', 'entire_maze'],
						   default='entire_maze', help='macro actions')
	arg_parser.add_argument('-m', '--mode', type=str, choices=['train', 'test'],
						   default='train', help='training or testing mode')
	arg_parser.add_argument('-e', '--environment', type=str, choices=['forum','room', 'navigation', 'maze'],
						   default='maze', help='experimental environment')
	arg_parser.add_argument('-v', '--validation', type=str, choices=['yes','no'],
						   default='no', help='whether or not to calculate validation error during training')

	args = arg_parser.parse_args()

	main(args.direction, args.mode, args.environment, args.validation)
