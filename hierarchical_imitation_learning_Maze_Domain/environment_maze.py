# Hoang M. Le
# 
# California Institute of Technology
# hmle@caltech.edu
# 
# ===================================================================================================================

from __future__ import absolute_import

import json
import re
import random
import numpy as np


from collections import Sequence
from time import sleep
from numpy import log
import Tkinter as tk
import six

ENV = 'maze'

ENV_GOAL_REWARD = 100
ENV_ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
HORIZON = 100

# Rewards
ACT = -1
OBSTACLE = -100
GOAL = 100

HALLWAY = [(4,8), (12,8), (8,4), (8,12)]

def parse_clients_args(args_clients):
	"""
	Return an array of tuples (ip, port) extracted from ip:port string
	:param args_clients:
	:return:
	"""
	return [str.split(str(client), ':') for client in args_clients]

def visualize_training(visualizer, step, rewards, tag='Training'):
	visualizer.add_entry(step, '%s/reward per episode' % tag, sum(rewards))
	visualizer.add_entry(step, '%s/max.reward' % tag, max(rewards))
	visualizer.add_entry(step, '%s/min.reward' % tag, min(rewards))
	visualizer.add_entry(step, '%s/actions per episode' % tag, len(rewards)-1)

#random_seed = 985 # for reproducibility
#random.seed(985)  # for reproducibility

class MazeNavigationStateBuilder(object):
	"""
	StateBuilder are object that map environment state into another representation.

	"""
	GRAY_PALETTE = {
		'x': 1.0, # lava, obstacle
		'o': 0.0, # stone, land
		'g': 0.5, # goal
		'a': 0.4, # agent
		'w': -0.5 # visited block ?
	}

	def __init__(self, gray=True):
		self._gray = bool(gray)

	def build(self, environment):
		#print "problem here"
		#raise NotImplementedError()
		if self._gray:
			buffer_shape = (environment.width, environment.height)
			buffer = np.zeros(buffer_shape, dtype = np.float32)
			for x in range(environment.width):
				for y in range(environment.height):
					buffer[x,y] = self.GRAY_PALETTE[environment._world[x,y]]

			# If alive, cell value for agent will be 0.2, else mix it with either the lava or the goal
			buffer[environment._agent_loc[0], environment._agent_loc[1]] = 0.5*( self.GRAY_PALETTE['a'] + buffer[environment._agent_loc[0], environment._agent_loc[1]]) 
			return buffer
		if not self._gray:
			buffer_shape = (environment.width, environment.height,3)
			#buffer_shape = (4,environment.width, environment.height)
			buffer = np.zeros(buffer_shape, dtype = np.float32)
			# agent layer:
			buffer[environment._agent_loc[0], environment._agent_loc[1],0] = 0.4 
			# obstacle layer:
			for x in range(environment.width):
				for y in range(environment.height):
					if environment._world[x,y] == 'x':
						buffer[x,y,1] = self.GRAY_PALETTE[environment._world[x,y]]
					elif environment._world[x,y] == 'w':
						buffer[x,y,1] = self.GRAY_PALETTE[environment._world[x,y]] # add the goo obstacles into the 4th channel
					elif environment._world[x,y] == 'g':
						buffer[x,y,2] = -1.0 ## Goal cell
			return buffer




	#def __call__(self, *args, **kwargs):
	#    return self.build(*args)
	def __call__(self, *args, **kwargs):
		assert isinstance(args[0], MazeNavigationEnvironment), 'provided argument should inherit from MazeNavigationEnvironment'
		return self.build(*args)

class MazeNavigationEnvironment(object):
	"""
	Create an environment for 2D navigation task
	"""

	def __init__(self,
			   state_builder,
			   actions=ENV_ACTIONS,
			   rendering=True, randomized_door = True,
			   stochastic_dynamic=True, map_id = None, horizon = HORIZON, setting = ENV):
		assert isinstance(state_builder, MazeNavigationStateBuilder)

		self._setting = setting
		global ENV_BOARD_SHAPE
		if self._setting == 'maze':
			ENV_BOARD_SHAPE = (17,17)

		self._width = ENV_BOARD_SHAPE[0]
		self._height = ENV_BOARD_SHAPE[1]
		self._map_id = map_id
		self._horizon = horizon
		

		# Load the world maps into a dictionary
		if self._setting == 'maze':
			self._maps_dict = np.load('maps_16rooms_landmarks.npy').item()


		self.stochastic_dynamic = stochastic_dynamic
		self.randomized_door = randomized_door

		assert self._map_id is not None, "map_id cannot be None"
		if self._map_id is not None:
			#self._world = self._maps_dict[self._map_id].copy()
			self._landmark = self._maps_dict[self._map_id]

		self._world = np.chararray((ENV_BOARD_SHAPE[0], ENV_BOARD_SHAPE[1]))
		self._world[:] = 'o'
		for row in range(5):
			for col in range(ENV_BOARD_SHAPE[0]):
				self._world[col, row*4] = 'x'
		for col in range(5):
			for row in range(ENV_BOARD_SHAPE[1]):
				self._world[col*4, row] = 'x'
		self._agent_loc = self._landmark[0]
		self._goal_loc = self._landmark[1]
		self._walls_for_door = self._landmark[2]
		
		self._world[self._goal_loc[0], self._goal_loc[1]] = 'g'
		self._world[self._agent_loc[0], self._agent_loc[1]] = 'w' # was here


		self._valid_start = []
		self._valid_start.append((self._agent_loc[0], self._agent_loc[1]))
		if self.randomized_door:
			random_doors_open = []
			for item in self._walls_for_door:
				if item[0] % 4 == 0:
					assert item[1] % 4 != 0
					random_coordinate = random.sample([item[1]-1, item[1], item[1]+1], 1)[0]
					random_doors_open.append((item[0], random_coordinate))
				elif item[1] % 4 == 0:
					assert item[0] % 4 != 0
					random_coordinate = random.sample([item[0]-1, item[0], item[0]+1], 1)[0]
					random_doors_open.append((random_coordinate, item[1]))

			for door in random_doors_open:
				self._world[door[0], door[1]] = 'o'
		else:		
			for door in self._walls_for_door:
				self._world[door[0], door[1]] = 'o'
		
		## Initialize the agent location
		self._agent_loc = self._randomize_initial_position()		

		self._user_defined_builder = state_builder
		
		self._reward = 0
		self._done = False
		self._state = None 
		
		self._actions = actions
		assert actions is not None, "actions cannot be None"
		assert isinstance(actions, Sequence), "actions should be an iterable object"
		assert len(actions) > 0, "len(actions) should be > 0"

		self._rendering = rendering
		
		### Setting up the world
		#self._world = None
		assert np.shape(self._world) == (self._width, self._height), 'board size mismatch'

		self._status = "initial"
	
		self._previous_action = None
		self._last_frame = None
		self._action_count = None
		self._end_result = None

		### Hoang: add several variables to check if the episode is over
		self._done = False
		
		self._hit = False

		# -- set up the python-side drawing -- #
		self.canvas = None
		self.root = None
		if self._rendering:
			scale = 30
			curr_radius = 0.2
			world_x = self._width
			world_y = self._height
			self.root = tk.Tk()
			self.root.wm_title("Maze-environment")
			self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
			self.canvas.grid()


	def render(self):
		if self.canvas is None or self.root is None:
			return
		self.canvas.delete("all")
		scale = 30
		curr_radius = 0.2
		world_x = self._width
		world_y = self._height

		for x in range(world_x):
			for y in range(world_y):
				s = "%d:%d" % (x,y)
				if self._world[x,y] == 'x':
					fill_color = "#b22222" # obstacles
					#fill_color = "#696969" # gray obstacles?
					#fill_color = "#000"
				elif self._world[x,y] == 'o':
					fill_color = "#000"
					#fill_color = "#f8f8ff"
					#fill_color = "#bebebe"
				elif self._world[x,y] == 'w':
					fill_color = "#20b2aa"
					#fill_color = "#9acd32" ## yellow green
					#fill_color = "#f0fff0" ## honeydew
					#fill_color = "#000"
				elif self._world[x,y] == 'g':
					fill_color = "#ffd700" #gold
				#self.canvas.create_rectangle( (world_x-1-x)*scale, (world_y-1-y)*scale, (world_x-1-x+1)*scale, (world_y-1-y+1)*scale, outline="#fff", fill=fill_color)
				self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill=fill_color)
		curr_x = self._agent_loc[0]
		curr_y = self._agent_loc[1]
		#print curr_x, curr_y
		if curr_x is not None and curr_y is not None:
			self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
									 (curr_y + 0.5 - curr_radius ) * scale, 
									 (curr_x + 0.5 + curr_radius ) * scale, 
									 (curr_y + 0.5 + curr_radius ) * scale, 
									 outline="#fff", fill="#fff" )
		self.root.update()

	@property
	def state(self):
		return self._user_defined_builder.build(self)
	@property
	def agent_loc(self):
		return (self._agent_loc[0], self._agent_loc[1])
	@property
	def width(self):
		return self._width
	@property
	def height(self):
		return self._height
	@property
	def available_actions(self):
		return len(self._actions)
	@property
	def previous_action(self):
		return self._previous_action
	@property
	def action_count(self):
		return self._action_count
	@property
	def reward(self):
		return self._reward
	@property
	def done(self):
		"""
		Done if we have reached the goal or died or out of time
		"""
		#return super(CliffWalkEnvironment, self).done
		return self._done
	@property
	def map_id(self):
		return self._map_id

	@property
	def goal_loc(self):
		self._goal_loc = (np.where(self._world == 'g')[0][0], np.where(self._world == 'g')[1][0])
		return self._goal_loc
	@property
	def agent_loc(self):
		return (self._agent_loc[0], self._agent_loc[1])
	@property
	def in_goo(self):
		return self._world[self._agent_loc[0], self._agent_loc[1]] == 'w'


	def identify_location_room(self, loc):
		if loc[0] < 8 and loc[1] < 8:
			return 1
		elif loc[0] > 8 and loc[1] < 8:
			return 2
		elif loc[0] < 8 and loc[1] > 8:
			return 3
		elif loc[0] > 8  and loc[1] >8:
			return 4
		elif (loc[0], loc[1]) in HALLWAY:
			return 0
		else:
			return -1 # basically invalid

	def identify_agent_room(self):
		if self._agent_loc[0] < 8 and self._agent_loc[1] < 8:
			return 1
		elif self._agent_loc[0] > 8 and self._agent_loc[1] < 8:
			return 2
		elif self._agent_loc[0] < 8 and self._agent_loc[1] > 8:
			return 3
		elif self._agent_loc[0] > 8  and self._agent_loc[1] >8:
			return 4
		elif (self._agent_loc[0], self._agent_loc[1]) in HALLWAY:
			return 0
		else:
			return -1 # basically invalid

	def identify_goal_room(self):
		(np.where(self._world == 'g')[0][0], np.where(self._world == 'g')[1][0])
		if self._goal_loc[0] < 8 and self._goal_loc[1] < 8:
			return 1
		elif self._goal_loc[0] > 8 and self._goal_loc[1] < 8:
			return 2
		elif self._goal_loc[0] < 8 and self._goal_loc[1] > 8:
			return 3
		elif self._goal_loc[0] > 8  and self._goal_loc[1] >8:
			return 4
		elif (self._goal_loc[0], self._goal_loc[1]) in HALLWAY:
			return 0 # this should never happen
		else:
			return -1 # basically invalid


	def _identify_starting_position(self, world_map):
		pos = (1000, 1)
		for i in range(1,7):
			if world_map[i,1] == 'a':
				pos = (i,1)
		return [int(pos[0]), int(pos[1])]


	def _randomize_initial_position(self):

		i = np.random.choice(range(len(self._valid_start)))
		pos = self._valid_start[i]

		return [int(pos[0]), int(pos[1])]

	def change_map_and_reset_with_given_start(self, map_id, start_positions):
		self._map_id = map_id

		# after that, it is pretty much exactly the same as reset
		self._reward = 0.
		self._done = False
		self._state = None

		self._previous_action = None
		self._action_count = 0
		self._end_result = None

		assert self._map_id is not None, "map_id cannot be None, this is wrong"
		if self._map_id is not None:
			self._world = self._maps_dict[self._map_id]
			self._valid_start = start_positions
			self._agent_loc = self._randomize_initial_position()

		return self.state

	def change_map_and_reset(self, map_id, easy_start = False):
		"""
		:first thing is to change the world generator:
		"""
		self._map_id = map_id

		# after that, it is pretty much exactly the same as reset
		self._reward = 0.
		self._done = False
		self._state = None

		self._previous_action = None
		self._action_count = 0
		self._end_result = None

		assert self._map_id is not None, "map_id cannot be None, this is wrong"

		if self._map_id is not None:
			#self._world = self._maps_dict[self._map_id].copy()
			self._landmark = self._maps_dict[self._map_id]

		self._world = np.chararray((ENV_BOARD_SHAPE[0], ENV_BOARD_SHAPE[1]))
		self._world[:] = 'o'
		for row in range(5):
			for col in range(ENV_BOARD_SHAPE[0]):
				self._world[col, row*4] = 'x'
		for col in range(5):
			for row in range(ENV_BOARD_SHAPE[1]):
				self._world[col*4, row] = 'x'
		self._agent_loc = self._landmark[0]
		self._goal_loc = self._landmark[1]
		self._walls_for_door = self._landmark[2]
		
		self._world[self._goal_loc[0], self._goal_loc[1]] = 'g'
		self._world[self._agent_loc[0], self._agent_loc[1]] = 'w' # was here


		self._valid_start = []
		self._valid_start.append((self._agent_loc[0], self._agent_loc[1]))
		if self.randomized_door:
			random_doors_open = []
			for item in self._walls_for_door:
				if item[0] % 4 == 0:
					assert item[1] % 4 != 0
					random_coordinate = random.sample([item[1]-1, item[1], item[1]+1], 1)[0]
					random_doors_open.append((item[0], random_coordinate))
				elif item[1] % 4 == 0:
					assert item[0] % 4 != 0
					random_coordinate = random.sample([item[0]-1, item[0], item[0]+1], 1)[0]
					random_doors_open.append((random_coordinate, item[1]))

			for door in random_doors_open:
				self._world[door[0], door[1]] = 'o'
		else:		
			for door in self._walls_for_door:
				self._world[door[0], door[1]] = 'o'
		
		## Initialize the agent location
		self._agent_loc = self._randomize_initial_position()		

		return self.state

	def reset(self):
		"""
		:return:
		"""
		self._reward = 0.
		self._done = False
		self._state = None


		self._previous_action = None
		self._action_count = 0
		self._end_result = None

		if self._map_id is not None:
			self._world = self._maps_dict[self._map_id].copy()
			self._agent_loc = self._randomize_initial_position()

		# wait for mission to begin
		return self.state

	def get_valid_start(self):
		valid = []
		for i in range(0,16):
			for j in range(0,16):
				if self._world[i,j] == 'o':
					valid.append((i,j))

		self._valid_start = valid

	def _update(self, action):
		#print self._agent_loc[1]
		if not self.stochastic_dynamic:
			old_loc = (self._agent_loc[0], self._agent_loc[1])
			if action == "movenorth 1":
				#self._agent_loc[1] -= 1
				new_loc = (old_loc[0], old_loc[1]-1)
			elif action == "movesouth 1":
				#self._agent_loc[1] += 1
				new_loc = (old_loc[0], old_loc[1]+1)
			elif action == "movewest 1":
				#self._agent_loc[0] -= 1
				new_loc = (old_loc[0]-1, old_loc[1])
			elif action == "moveeast 1":
				#self._agent_loc[0] += 1 ### Hoang: be careful about orientation and how it differs from Malmo
				new_loc = (old_loc[0]+1, old_loc[1])
			else:
				print "Illegal action"
			if self._world[old_loc[0], old_loc[1]] != 'x':
				self._agent_loc[0] = new_loc[0]
				self._agent_loc[1] = new_loc[1]
			else:
				if new_loc[0]<0 or new_loc[0] > 16 or new_loc[1]<0 or new_loc[1] >16:
					self._agent_loc[0] = old_loc[0]
					self._agent_loc[1] = old_loc[1]					
				elif self._world[new_loc[0], new_loc[1]] == 'w':
					self._agent_loc[0] = new_loc[0]
					self._agent_loc[1] = new_loc[1]
				else:
					self._agent_loc[0] = old_loc[0]
					self._agent_loc[1] = old_loc[1]

			if self._world[self._agent_loc[0], self._agent_loc[1]] != 'g' and self._world[self._agent_loc[0], self._agent_loc[1]] != 'x':
				self._world[self._agent_loc[0], self._agent_loc[1]] = 'w' # was here

		else:
			if action == "movenorth 1":
				prob_vec = [0.85, 0.05, 0.05, 0.05]
			elif action == "movesouth 1":
				prob_vec = [0.05, 0.85, 0.05, 0.05]
			elif action == "movewest 1":
				prob_vec = [0.05, 0.05, 0.85, 0.05]
			elif action == "moveeast 1":
				prob_vec = [0.05, 0.05, 0.05, 0.85]
			action_sample = np.random.choice(range(4), p=prob_vec)
			directions = [(0,-1), (0,1), (-1,0), (1,0)]
			new_loc = (self._agent_loc[0] + directions[action_sample][0], self._agent_loc[1] + directions[action_sample][1])
			if self._world[new_loc[0], new_loc[1]] != 'x':
				self._agent_loc[0] = new_loc[0]
				self._agent_loc[1] = new_loc[1]
			#self._agent_loc[0] = self._agent_loc[0] + directions[action_sample][0]
			#self._agent_loc[1] = self._agent_loc[1] + directions[action_sample][1]
			if self._world[self._agent_loc[0], self._agent_loc[1]] != 'g':
				self._world[self._agent_loc[0], self._agent_loc[1]] = 'w' # was here

		if self._world[self._agent_loc[0], self._agent_loc[1]] == 'o':
			self._status = "alive"
			self._reward = -0.01
		elif self._world[self._agent_loc[0], self._agent_loc[1]] == 'x':
			self._status = "trapped"
			self._reward = -0.01
			#self._done = True
		elif self._world[self._agent_loc[0], self._agent_loc[1]] == 'w':
			self._status = "visited"
			self._reward = -0.01
		elif self._world[self._agent_loc[0], self._agent_loc[1]] == 'g':
			self._status = "succeed"
			self._reward = 1
			self._done = True
		else:
			print "there is something wrong with mapping location to status"

		#self._world_obs = self._world.copy()
		#self._world_obs[self._agent_loc[0], self._agent_loc[1]] = 'a'

	def abstract_state_loc(self):
		return [int(self._agent_loc[0]/4), int(self._agent_loc[1]/4)]
	

	def do(self, action_id):
		assert 0 <= action_id <= self.available_actions, \
			"action %d is not valid (should be in [0, %d[)" % (action_id,
															   self.available_actions)
		action = self._actions[action_id]
		assert isinstance(action, six.string_types)


		self._update(action)
		self._previous_action = action
		self._action_count += 1
		if self._action_count >= self._horizon:
			self._done = True

		#self._await_next_obs()
		#return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done
		return self.state, self.reward, self.done
	
	"""
	def do(self, action):
		state, reward, done = super(CliffWalkEnvironment, self).do(action)
		return state, reward, self.done
	"""
	def is_valid(self, world_state):
		""" Pig Chase Environment is valid if the the board and entities are present """
		"""
		if super(CliffWalkEnvironment, self).is_valid(world_state):
			obs = json.loads(world_state.observations[-1].text)

			# Check we have entities
			return (ENV_ENTITIES in obs) and (ENV_BOARD in obs)
		return False   
		"""
		return True #hoang: override it for now
