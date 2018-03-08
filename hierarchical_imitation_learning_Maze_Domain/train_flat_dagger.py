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

SUMMARY_NAME = 'fda_200WarmStart_1pass_1000maps_randomDoor_run4'
# General parameters
NUM_EPISODES = 1000
TEST_EPISODES = 1000

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

# NN parameters
TRAIN_HIST_SIZE = 100000
EXPERIENCE_MEMORY = 1000000
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

class NN:
	def __init__(self, input_shape):
		with tf.device('/gpu:0'):
			inputs = Input(shape = input_shape)

			x = Conv2D(32, kernel_size = 3, strides = (1,1), padding = 'same', data_format = 'channels_last', activation = 'relu')(inputs)
			x = Conv2D(32, kernel_size = 3, strides = (1,1),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			x = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(x)
			x = Conv2D(64,(3,3),  activation = 'relu')(x)
			x = MaxPooling2D(pool_size = (2,2))(x)
			x = Dropout(0.5)(x)

			#x = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(x)
			#x = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(x)
			#x = MaxPooling2D(pool_size = (2,2))(x)
			#x = Dropout(0.25)(x)


			conv_out = Flatten()(x)
			#y = Dropout(0.5)(conv_out)
			y = Dense(256, activation = 'relu')(conv_out)
			y = Dropout(0.5)(y)
			#y = Dense(128, activation = 'relu')(y)
			#y = Dropout(0.5)(y)
			action_outputs = Dense(OUTPUT_DIM, activation = 'softmax', name = 'action_outputs')(y)

			#z = Dropout(0.5)(conv_out)
			#z = Dense(64, activation = 'relu')(conv_out)
			#z = Dropout(0.5)(z)
			#goal_output = Dense(1, activation = 'sigmoid', name = 'goal_output')(z)

			self.model = Model(inputs = inputs, outputs = action_outputs)
			self.model.compile(loss = 'kullback_leibler_divergence', 
								optimizer = Adam(lr = LEARNING_RATE))
		#self.model.compile(loss = {'action_outputs':'categorical_crossentropy', 'goal_output':'binary_crossentropy'}, 
		#					optimizer = Adam(lr = LEARNING_RATE), loss_weights={'action_outputs':1., 'goal_output':10.})
		# Print out model dimensions
		logger.warning('Model input dim: ' + str(self.model.layers[0].input_shape))
		for l in self.model.layers:
			logger.warning('Output dim: ' + str(l.output_shape))
		# store history of length TRAIN_HIST_SIZE
		#time.sleep(30)
		self.replay_hist = [None] * TRAIN_HIST_SIZE
		self.ind = 0

		self.input_shape = input_shape
		self._history = LossHistory()


	def collect(self, processed, expert_a):
		if processed is not None:
			self.replay_hist[self.ind] = (processed.astype(np.float32), expert_a.astype(np.float32))
			self.ind = (self.ind + 1) % TRAIN_HIST_SIZE

	def end_collect(self):
		try:
			return self.train()
		except:
			return

	def _replay_to_train(self, replay_data_batched):
		batch_size = len(replay_data_batched)

		# get current q predictions
		x = [replay_data_batched[i][0] for i in xrange(batch_size)]
		x = np.reshape(x, (batch_size, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH))
		
		#q_pred = self.predict(x, batch_size=batch_size)
		expert_a = [replay_data_batched[i][1] for i in xrange(batch_size)]
		expert_a = np.reshape(expert_a, (batch_size,4))

		goal = [replay_data_batched[i][2] for i in xrange(batch_size)]
		goal = np.reshape(goal, (batch_size,1))

		# return training data
		return x, expert_a, goal

	def train(self):
		# if not reached TRAIN_HIST_SIZE yet, then get the number of samples
		self._num_valid = self.ind if self.replay_hist[-1] == None else TRAIN_HIST_SIZE
		#self._samples = np.random.choice(range(self._num_valid), size=BATCH_SIZE)
		try:
			#self._samples = random.sample(range(self._num_valid), BATCH_SIZE)
			self._samples = range(self._num_valid)
			BATCH_SIZE = len(self._samples)
		except:
			self._samples = range(self._num_valid) + [0] * (BATCH_SIZE - len(range(self._num_valid)))

		# convert replay data to trainable data
		self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
		self._train_x = np.reshape([self._selected_replay_data[i][0] for i in xrange(BATCH_SIZE)],
									(BATCH_SIZE,) + self.input_shape)
		self._train_y = np.reshape([self._selected_replay_data[i][1] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,4))
		#self._train_g = np.reshape([self._selected_replay_data[i][2] for i in xrange(BATCH_SIZE)],(BATCH_SIZE,1))

		self.model.fit(self._train_x, self._train_y, batch_size = 32, epochs = 1, callbacks = [self._history])
		#logger.info("Loss: " + str(self._history.losses))
		return self._history.losses

	def predict(self, x, batch_size=1):
		"""predict on (a batch of) x"""
		return self.model.predict(x, batch_size=batch_size, verbose=0)


# Agent logic
class Agent(Visualizable):
	def __init__(self, mode, env, direction):
		if env == 'maze':
			self.input_shape = (17, 17,CHANNELS)
		self.model = NN(self.input_shape)

		self.direction = direction

		#self.expert = expert
		self.mode = mode
		if mode == 'train':
			#logdir = path.join('summary/flat_dagger/warm_start_300episodes_lr000025_run1', datetime.utcnow().isoformat()) ## subject to change
			logdir = path.join('summary/flat_dagger/', SUMMARY_NAME) ## subject to change
			self._visualizer = TensorboardVisualizer()
			self._visualizer.initialize(logdir,None)
		if mode == 'test':
			if self.direction == 'north':
				fileName= 'saved_networks/dagger_option/go_north_floorklLoss_c_lr001_12.h5'
			elif self.direction == 'south':
				fileName= 'saved_networks/dagger_option/go_south_floorklLoss_c_lr001_12.h5'
			elif self.direction == 'west':
				fileName= 'saved_networks/dagger_option/go_west_floorklLoss_c_lr001_12.h5'
			elif self.direction == 'east':
				fileName= 'saved_networks/dagger_option/go_east_floorklLoss_c_lr001_12.h5'
			elif self.direction ==  'entire_maze':
				fileName= 'saved_networks/go_entire_maze_floor_acrossRoom_start_3264256_3channels_80.h5'
			#fileName = 'saved_networks/dagger_option/go_west_floorklLoss_c_lr001_12.h5'
			self.load_model(fileName)
		
		self.total_steps = 0

		self._stats_loss = []
		self._stats_rewards = []
		self._stats_val_rewards = 0
		self._success = 0
		self._stats_success = [] 
		self._stats_performance = []
		if self.direction == 'north':
			self.goal_loc = [(4,7), (12,7)]
		elif self.direction == 'south':
			self.goal_loc = [(4,9), (12,9)]
		elif self.direction == 'west':
			self.goal_loc = [(7,4), (7,12)]
		elif self.direction == 'east':
			self.goal_loc = [(9,4), (9,12)]
		elif self.direction == 'entire_maze':
			self.goal_loc = [(-1,-1)] ## doesnt exist

		self.experience = [None] * EXPERIENCE_MEMORY
		self.ind = 0

		self._expert_act = True
		self.warmstart = True

	def turn_off_warmstart(self):
		self.warmstart = False

	def collect_experience(self, prev_state, agent_action, reward, next_state, expert_advice):
		self.experience[self.ind] = (prev_state.astype(np.float32), agent_action, reward, next_state.astype(np.float32), expert_advice.astype(np.float32))
		self.ind = (self.ind + 1) % EXPERIENCE_MEMORY

	def save_experience(self, fileName):
		with open(fileName, 'wb') as f:
			pickle.dump(self.experience, f, protocol=pickle.HIGHEST_PROTOCOL)

	def save_success_record(self, fileName):
		with open(fileName, 'wb') as f:
			pickle.dump(self._stats_performance, f, protocol=pickle.HIGHEST_PROTOCOL)

	def change_expert(self, dictionary):
		self.expert = dictionary
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
		value = value_iteration(maze)
		policy = best_policy(maze, value)
		self.expert= {}
		self.expert_value = value

		
		for row in range(17):
		    for col in range(17):
		        if policy[(col, row)] == (0,-1):
		            self.expert[(col,row)] = 0 #N
		        elif policy[(col, row)] == (0,1):
		            self.expert[(col,row)] = 1 #S
		        elif policy[(col, row)] == (-1,0):
		            self.expert[(col,row)] = 2 #W
		        elif policy[(col, row)] == (1,0):
		            self.expert[(col,row)] = 3 #E


	def act(self, world_state, agent_host, prev_r):
		self._state = np.reshape(world_state, (1,)+self.input_shape)

		self._prev_x = agent_host._agent_loc[0]
		self._prev_z = agent_host._agent_loc[1]
		#self._table_loc = '%s:%s' % (self._prev_x,self._prev_z)
		self._table_loc = (self._prev_x,self._prev_z)
		#print self.expert[table_loc]

		#assert self.expert[self._table_loc] is not None
		if agent_host._world[self._prev_x, self._prev_z] == 'x':
			## dealing with trap state
			possible_actions = [(0,-1), (0,1), (-1,0), (1,0)]
			best_direction = (0,0)
			max_value = -1.0
			for action in range(4):
				direction = possible_actions[action]
				state = (self._prev_x+direction[0], self._prev_z+direction[1])
				if state[0]>=0 and state[0]<=16 and state[1]>=0 and state[1]<=16 and agent_host._world[state[0], state[1]] == 'w':
					if self.expert_value[state] > max_value:
						max_value = self.expert_value[state]
						best_direction = direction
			expert_a = possible_actions.index(best_direction)
		else:
			assert self.expert[self._table_loc] is not None
			expert_a = self.expert[self._table_loc]
		
		self.expert_a = np.zeros((1,len(ACTIONS)))
		self.expert_a[0,expert_a] = 1.

		# collect into experience memory
		if self.warmstart:
			#self._a = self.sample(self.expert_a[0])
			self._a = expert_a
		else:
			self._prob_pred = self.model.predict(self._state)[0]
			self._a = self.sample(self._prob_pred)

		self.model.collect(self._state, self.expert_a)
		self.total_steps += 1

		agent_host._update(ACTIONS[self._a])

		## after executing the action, collect this transition into the separate memory bank, to aid the reinforcement learning process
		reward = agent_host.reward
		next_state = agent_host.state
		self.collect_experience(world_state, self._a, reward, next_state, self.expert_a)

		return prev_r

	def run(self, agent_host):
		"""run the agent on the world"""

		self._total_reward = 0
		self._prev_r = 0
		#tol = 0.01
		self._step = 0 #step count

		self.expert_a = None

		self._world_state = agent_host.state

		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1) # (let the Mod reset)

		# take first action
		#try:
		self._total_reward += self.act(self._world_state,agent_host,self._prev_r)
		self._step +=1
		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1)

		# main loop:
		while not agent_host.done:
			self._world_state = agent_host.state
			self._prev_r = agent_host.reward

			if self._step<HORIZON:
				#try:
				self._total_reward += self.act(self._world_state, agent_host, self._prev_r)
				self._step += 1
				if agent_host._rendering:
					agent_host.render()
					time.sleep(0.1)
				#except:
				#	print "something wrong, should check"
				#	agent_host._done = True
			else:
				break
		self._total_reward += agent_host.reward
		
		if self.mode == 'train' and self.total_steps>=100:
			loss = self.model.end_collect()
			print loss[-1]
			self._stats_loss.append(sum(loss)/len(loss))

		return self._total_reward

	def act_test(self, world_state, agent_host, prev_r):
		self._state = np.reshape(world_state, (1,)+self.input_shape)

		self._prob_pred = self.model.predict(self._state)[0]
		self._a = self.sample(self._prob_pred)

		agent_host._update(ACTIONS[self._a])

		## after executing the action, collect this transition into the separate memory bank, to aid the reinforcement learning process
		reward = agent_host.reward
		return prev_r

	def run_test(self, agent_host):
		"""run the agent on the world"""

		self._total_reward = 0
		self._prev_r = 0
		#tol = 0.01
		self._step = 0 #step count

		self._world_state = agent_host.state

		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1) # (let the Mod reset)


		# take first action
		#try:
		self._total_reward += self.act_test(self._world_state,agent_host,self._prev_r)
		self._step +=1
		if agent_host._rendering:
			agent_host.render()
			time.sleep(0.1)

		# main loop:
		while not agent_host.done:

			# wait for the position to have changed and a reward received
			#logger.debug('Waiting for data...')

			self._world_state = agent_host.state
			#prev_r = sum(r.getValue() for r in world_state.rewards)
			self._prev_r = agent_host.reward

			if self._step<HORIZON:
				# act
				#try:
				self._total_reward += self.act_test(self._world_state, agent_host, self._prev_r)
				self._step += 1
				if agent_host._rendering:
					agent_host.render()
					time.sleep(0.1)
				#except:
				#	print "something wrong during testing"
				#	agent_host._done = True
					#time.sleep(100)
			else:
				break
				#time.sleep(0.5) # (let the Mod reset)

		# process final reward
		logger.debug("Final reward: %d" % agent_host.reward)
		if agent_host.reward > 0:
			self._success = 1
		print 
		self._total_reward += agent_host.reward

		self._stats_rewards.append(self._total_reward)

		# final update
		self._stats_success.append(self._success)
		self._stats_performance.append((self._success, self.ind))
		self._success = 0

		#agent_host.reset()

		return self._total_reward


	def inject_summaries_with_val(self, idx, val_reward):
		self._stats_val_rewards = val_reward
		if self.mode == 'train':
			if len(self._stats_loss) > 0:
				self.visualize(idx, "episode loss",
							   np.asscalar(np.mean(self._stats_loss)))

			if len(self._stats_rewards) > 0:
				self.visualize(idx, "episode reward",
							   np.asscalar(np.mean(self._stats_rewards)))
			self.visualize(idx, "validation reward", self._stats_val_rewards)
			# Reset
			self._stats_loss = []
			self._stats_rewards = []

	def inject_summaries(self, idx):
		if self.mode == 'train':
			if len(self._stats_loss) > 0:
				self.visualize(idx, "episode loss",
							   np.asscalar(np.mean(self._stats_loss)))

			if len(self._stats_rewards) > 0:
				self.visualize(idx, "episode reward",
							   np.asscalar(np.mean(self._stats_rewards)))
			self.visualize(idx, "expert labels for flat DAgger", self.ind)
			## Showing the success rate for the trailing 20 training episode
			#if not self.warmstart:
			if len(self._stats_success) >100:
				self.visualize(idx, "episode success indicator", np.asscalar(np.mean(self._stats_success[-100:])))
				self.visualize(self.ind, "learning curve",np.asscalar(np.mean(self._stats_success[-100:])))
			else:
				self.visualize(idx, "episode success indicator", np.asscalar(np.mean(self._stats_success)))
				self.visualize(self.ind, "learning curve",np.asscalar(np.mean(self._stats_success)))

			# Reset
			self._stats_loss = []
			self._stats_rewards = []
	def load_model(self, fileName):
		self.model.model.load_weights(fileName)

def main(macro_action, train_or_test, environment, validation):
	global NUM_EPISODES

	direction = macro_action
	mode = train_or_test

	MODEL_NAME = 'go_'+direction+'_'+environment+'_acrossRoom_start_3264256_3channels_'


	agent_host = MazeNavigationEnvironment(MazeNavigationStateBuilder(gray = False),
									rendering = False, randomized_door = True, stochastic_dynamic=False, map_id=999, setting = environment)

	agent = Agent(mode, environment, direction)

	map_ids_to_use = range(1000,2000)

	if mode == 'test':
		NUM_EPISODES = TEST_EPISODES
	
	#for i in range(1, NUM_EPISODES + 1):
	for i in six.moves.range(1,NUM_EPISODES+1):

		logger.info("\nMission %d of %d:" % ( i, NUM_EPISODES ))

		#logger.debug("Spinning up a new environment and expert")

		new_map_id = np.random.choice(range(1000))
		print 
		print "--------------------------------------------------------------------"
		logger.debug("Starting mission")
		
		agent_host.change_map_and_reset(new_map_id)
		agent.get_expert_policy(agent_host)

		#print "expert table number:", agent_host.map_id

		world_state = agent_host.state

		# -- run the agent in the world -- #
		if i >200:
			agent.turn_off_warmstart()
		
		cumulative_reward = agent.run(agent_host)

		#### RUN THE ACTUAL VALIDATION
		new_map_id = np.random.choice(map_ids_to_use)

		agent_host.change_map_and_reset(new_map_id)
		cumulative_reward = agent.run_test(agent_host)


		agent.inject_summaries(i)


		logger.info("Cumulative reward: " + str(cumulative_reward))

		if i % SAVE_MODEL_EVERY == 0 and mode == 'train':
			print "skip model saving"
			agent.save_success_record('summary/success_record/result_'+SUMMARY_NAME+'_'+str(int(i/SAVE_MODEL_EVERY))+'.pkl')
			#agent.model.model.save_weights('saved_networks/flat_dagger/'+MODEL_NAME+str(int(i/SAVE_MODEL_EVERY))+'.h5', overwrite = True)
			#agent.save_experience('saved_networks/saved_experience/'+MODEL_NAME+str(int(i/SAVE_MODEL_EVERY))+'.pkl')
			#agent.model.model.save_weights(MODEL_NAME, overwrite = True)

	if mode == 'test':
		logger.info("Overall success rate is:" + str(sum(agent._stats_success) *1.0/ len(agent._stats_success)))
	logger.warning("Done.")


if __name__ == '__main__':
	arg_parser = ArgumentParser('flat Dagger experiment')
	arg_parser.add_argument('-d', '--direction', type=str, choices=['north', 'south', 'west', 'east', 'entire_maze'],
						   default='entire_maze', help='macro actions')
	arg_parser.add_argument('-m', '--mode', type=str, choices=['train', 'test'],
						   default='train', help='training or testing mode')
	arg_parser.add_argument('-e', '--environment', type=str, choices=['forum','room', 'navigation', 'maze'],
						   default='maze', help='experimental environment')
	arg_parser.add_argument('-v', '--validation', type=str, choices=['yes','no'],
						   default='no', help='whether or not to calculate validation error during training')

	args = arg_parser.parse_args()

	#test_model_name

	main(args.direction, args.mode, args.environment, args.validation)
