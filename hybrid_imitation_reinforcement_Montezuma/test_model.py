# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# 
# Simple testing of trained subgoal models
# ===================================================================================================================

import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
from environment_atari import ALEEnvironment
from hybrid_rl_il_agent_atari import Agent
from hybrid_model_atari import Hdqn
from simple_net import Net
from PIL import Image
from tensorboard import TensorboardVisualizer
from os import path
import time

nb_Action = 8
# Constant defined here
maxStepsPerEpisode = 500
np.random.seed(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    #actionMap = [1, 2, 3, 4, 5, 11, 12] # testing: taking out no np action to see what happens
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    goalExplain = ['lower right ladder',  'key', 'lower right ladder', 'right door']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=True)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=True)
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    parser.add_argument("--load_weight", default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
    args = parser.parse_args()
    env = ALEEnvironment(args.game, args)
    
    # Initilize network and agent
    episodeCount = 0

    firstNet = Net()
    secondNet = Net()
    thirdNet = Net()
    fourthNet = Net()
    firstNet.loadWeight(0)
    secondNet.loadWeight(1)
    thirdNet.loadWeight(2)
    fourthNet.loadWeight(3)

    #for episode in range(80000):
    while episodeCount < 80000:
        print("\n\n### EPISODE "  + str(episodeCount) + "###")
        # Restart the game
        time.sleep(1)
        env.restart()
        episodeSteps = 0

        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            stateLastGoal = env.getStackedState()
            
            #goal = agent.selectGoal(stateLastGoal) #hoang: disable this for now, just go for the hard-coded sequence
            # incorporating the meta controller is very simple with only 4 meta actions
            goal = 0

            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = firstNet.selectMove(state, goal)

                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print
                #print action
                
                externalRewards = env.act(actionMap[action])
                time.sleep(0.01)

                episodeSteps += 1
                nextState = env.getStackedState()
                                                            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
            else:
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

            goal = 1
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = secondNet.selectMove(state, goal)
                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print
                #print action
                externalRewards = env.act(actionMap[action])
                time.sleep(0.01)
                episodeSteps += 1                
                nextState = env.getStackedState()

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])                
            else:
                break

            goal = 2
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = thirdNet.selectMove(state, goal)
                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print

                #print "action chosen:", action
                #print action
                externalRewards = env.act(actionMap[action])
                time.sleep(0.01)
                episodeSteps += 1
                #print episodeSteps
                nextState = env.getStackedState()
                                                
            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                
            else:
                break
            
            goal = 3
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = fourthNet.selectMove(state, goal)
                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print

                #print "action chosen:", action
                #print action
                externalRewards = env.act(actionMap[action])
                time.sleep(0.01)
                episodeSteps += 1
                #print episodeSteps
                nextState = env.getStackedState()

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                print('goal reached: ' + goalExplain[goal])
                #time.sleep(60)
                if goal == 3:
                    for i in range(15):
                        env.act(3)
                    for i in range(15):
                        env.act(0)
                    break
            else:
                break
        
        episodeCount += 1


if __name__ == "__main__":
    main()
