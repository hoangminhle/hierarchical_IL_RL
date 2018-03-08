# 
# Script from implementation of Russell And Norvig's "Artificial Intelligence - A Modern Approach"
# https://github.com/aimacode/aima-python
# 
# ===================================================================================================================

"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from utils import *
import numpy as np
import Tkinter as tk
from time import sleep


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        #grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))

class MazeMDP(MDP):
    def __init__(self, grid, terminals, init=(0,0), gamma=.9):
        #grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist = orientations, terminals = terminals, gamma = gamma)
        update(self, grid=grid, rows = len(grid), cols = len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[x][y]
                if grid[x][y] is not None:
                    self.states.add((x, y))
        #print "reward:"
        #print self.reward[(4,2)]
    def T(self, state, action):
        #if self.reward[state] == -0.01:
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
            (0.05, self.go(state, (0,-1))),
            (0.05, self.go(state, (0,1))),
            (0.05, self.go(state, (-1,0))),
            (0.05, self.go(state, (1,0)))]
        #else:
        #    return [(0.25, self.go(state, (0,-1))),
        #            (0.25, self.go(state, (0,1))),
        #            (0.25, self.go(state, (-1,0))),
        #            (0.25, self.go(state, (1,0)))]


    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(0, -1):'^', (0, 1):'v', (-1, 0):'<', (1, 0):'>', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s] for (p, s1) in T(s, pi[s])])
    return U


fig = GridMDP([[-0.04, -0.04, -0.04, +1],
                     [-0.04, None,  -0.04, -1],
                     [-0.04, -0.04, -0.04, -0.04]],
                    terminals=[(3, 2), (3, 1)])

def render(world, agent_loc):

    canvas.delete("all")

    for x in range(world_x):
        for y in range(world_y):
            s = "%d:%d" % (x,y)
            if world[x,y] == 'x':
                fill_color = "#b22222" # obstacles
                #fill_color = "#696969" # gray obstacles?
                #fill_color = "#000"
            elif world[x,y] == 'o':
                fill_color = "#000"
                #fill_color = "#f8f8ff"
                #fill_color = "#bebebe"
            elif world[x,y] == 'd':
                #fill_color = "#9acd32" ## yellow green
                #fill_color = "#f0fff0" ## honeydew
                fill_color = "#000"
            elif world[x,y] == 'g':
                fill_color = "#ffd700" #gold
            #self.canvas.create_rectangle( (world_x-1-x)*scale, (world_y-1-y)*scale, (world_x-1-x+1)*scale, (world_y-1-y+1)*scale, outline="#fff", fill=fill_color)
            canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill=fill_color)
    curr_x = agent_loc[0]
    curr_y = agent_loc[1]
    #print curr_x, curr_y
    if curr_x is not None and curr_y is not None:
        canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                 (curr_y + 0.5 - curr_radius ) * scale, 
                                 (curr_x + 0.5 + curr_radius ) * scale, 
                                 (curr_y + 0.5 + curr_radius ) * scale, 
                                 outline="#fff", fill="#fff" )
    root.update()

#policy = policy_iteration(fig)

#print fig.states
#print fig.to_grid(policy)
"""

#Setting up MDP
maps_dict = np.load('maps_16rooms_trial_locAndDoor.npy').item()

map_id = 1000
#random.seed(map_id)

landmark = maps_dict[map_id]

WIDTH = 17
HEIGHT = 17
world = np.chararray((WIDTH, HEIGHT))
world[:] = 'o'
for row in range(5):
    for col in range(WIDTH):
        world[col, row*4] = 'x'
for col in range(5):
    for row in range(HEIGHT):
        world[col*4, row] = 'x'
agent_loc = landmark[0]
goal_loc = landmark[1]
walls_for_door = landmark[2]

world[goal_loc[0], goal_loc[1]] = 'g'
world[agent_loc[0], agent_loc[1]] = 'a'


#random_doors_open = []
#for item in walls_for_door:
#    if item[0] % 4 == 0:
#        assert item[1] % 4 != 0
#        random_coordinate = random.sample([item[1]-1, item[1], item[1]+1], 1)[0]
#        random_doors_open.append((item[0], random_coordinate))
#    elif item[1] % 4 == 0:
#        assert item[0] % 4 != 0
#        random_coordinate = random.sample([item[0]-1, item[0], item[0]+1], 1)[0]
#        random_doors_open.append((random_coordinate, item[1]))


random_doors_open = walls_for_door
for door in random_doors_open:
    world[door[0], door[1]] = 'o'

print world

reward_description = []
terminal_description = []
for x in range(17):
    row_reward = []
    for y in range(17):
        if world[x, y] == 'o':
            row_reward.append(-0.01)
        elif world[x, y] == 'a':
            row_reward.append(-0.01)            
        elif world[x, y] == 'x':
            row_reward.append(-0.01)
            terminal_description.append((x,y))
        elif world[x, y] == 'g':
            row_reward.append(1)
            terminal_description.append((x,y))
    reward_description.append(row_reward)

print reward_description
print terminal_description

maze = MazeMDP(reward_description, terminal_description, init = agent_loc, gamma = 0.99)
#policy = policy_iteration(maze)

value = value_iteration(maze)
print "value function:"
print value
policy = best_policy(maze, value)

#optimal_actions =  maze.to_arrows(policy)

#print policy

policy_description = np.chararray((WIDTH, HEIGHT))
policy_description[:] = "."

for row in range(17):
    for col in range(17):
        if policy[(col, row)] == (0,-1):
            policy_description[col, row] = '<' ## reverse from usual
        elif policy[(col, row)] == (0,1):
            policy_description[col, row] = '>'
        elif policy[(col, row)] == (-1,0):
            policy_description[col, row] = '^'
        elif policy[(col, row)] == (1,0):
            policy_description[col, row] = 'v'

print policy_description

### run policy
#starting  = agent_loc
#print starting

#agent_loc = (15,15)

scale = 30
curr_radius = 0.2
world_x = WIDTH
world_y = HEIGHT

root = tk.Tk()
root.wm_title("Maze-environment")
canvas = tk.Canvas(root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()

render(world, agent_loc)
sleep(0.1)

print world[agent_loc[0], agent_loc[1]]
while agent_loc != goal_loc:
    if world[agent_loc[0], agent_loc[1]] == 'o' or world[agent_loc[0], agent_loc[1]] == 'a':
        #print "true here"
        chosen_action = policy[agent_loc]
        #prob_vector = [0.9,0.025,0.025,0.025,0.025]
        prob_vector = [1.0,0.0,0.0,0.0,0.0]
        choices = [chosen_action, (0,-1), (0,1), (-1,0), (1,0)]
        action_index = np.random.choice(range(5), p=prob_vector)
        action = choices[action_index]

        agent_loc = (agent_loc[0]+action[0], agent_loc[1]+action[1])
        render(world, agent_loc)
        sleep(0.1)
    elif world[agent_loc[0], agent_loc[1]] == 'x':
        action = random.choice([(0,-1), (0,1), (-1,0), (1,0)])
        agent_loc = (agent_loc[0]+action[0], agent_loc[1]+action[1])
        render(world, agent_loc)
        sleep(0.1)
    #else:
    #    print "something wrong"


sleep(2)
#print maze.states
"""