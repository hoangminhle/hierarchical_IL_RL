# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
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
    
    goalSuccessTrack = [deque(), deque(), deque(), deque()] # deque in python is linkedlist, list is actually an array
    goalSuccessCount = [0, 0, 0, 0]
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
    parser.add_argument("--test_mode", type=str2bool, default=False)
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    annealComplete = True
    saveExternalRewardScreen = True
    env = ALEEnvironment(args.game, args)
    
    # Initilize network and agent
    intrinsicRewardMonitor = 0
    externalRewardMonitor = 0
    totalIntrinsicReward = 0
    option_learned = False
    episodeCount = 0
    stepCount = 0

    firstNet = Net()
    secondNet = Net()
    thirdNet = Net()
    fourthNet = Net()
    firstNet.loadWeight(0)
    secondNet.loadWeight(1)
    thirdNet.loadWeight(2)
    fourthNet.loadWeight(3)

    #for episode in range(80000):
    while episodeCount < 80000 and stepCount < 1000000 and (not option_learned):
        print("\n\n### EPISODE "  + str(episodeCount) + "###")
        print("\n\n### STEPS "  + str(stepCount) + "###")
        # Restart the game
        time.sleep(2)
        env.restart()
        episodeSteps = 0


        # set goalNum to hardcoded subgoal
        lastGoal = -1
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode:
            #totalIntrinsicReward = 0
            totalExternalRewards = 0 # NOT SURE IF IT SHOULD BE CLEARED HERE!
            stateLastGoal = env.getStackedState()
            # nextState = stateLastGoal
            
            #goal = agent.selectGoal(stateLastGoal) #hoang: disable this for now, just go for the hard-coded sequence
            goal = 0
            if (len(goalSuccessTrack[goal]) > 100):
                firstElement = goalSuccessTrack[goal].popleft()
                goalSuccessCount[goal] -= firstElement
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = firstNet.selectMove(state, goal)

                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print
                #print action
                
                externalRewards = env.act(actionMap[action])
                time.sleep(0.05)
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                stepCount += 1
                episodeSteps += 1
                nextState = env.getStackedState()
                                                
                # Update data for visualization
                externalRewardMonitor += externalRewards

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                goalSuccessTrack[goal].append(0)
                break
            elif env.goalReached(goal):
                goalSuccessTrack[goal].append(1)
                goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                
                #lastGoal = goal
                # get key
                #if goal == 0:
                #    break
            else:
                goalSuccessTrack[goal].append(0)
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

            goal = 1
            if (len(goalSuccessTrack[goal]) > 100):
                firstElement = goalSuccessTrack[goal].popleft()
                goalSuccessCount[goal] -= firstElement
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = secondNet.selectMove(state, goal)
                print "episode step: ", episodeSteps
                print "action chosen:", actionExplain[action]
                print
                #print action
                externalRewards = env.act(actionMap[action])
                time.sleep(0.05)
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                stepCount += 1
                episodeSteps += 1                
                nextState = env.getStackedState()
                                                
                # Update data for visualization
                externalRewardMonitor += externalRewards

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                goalSuccessTrack[goal].append(0)
                break
            elif env.goalReached(goal):
                goalSuccessTrack[goal].append(1)
                goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                
                #lastGoal = goal
                # get key
                #if goal == 2:
                #    break
            else:
                goalSuccessTrack[goal].append(0)
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

            for i in range(20):
                env.act(actionMap[0]) # take a bunch of no ops


            goal = 2
            if (len(goalSuccessTrack[goal]) > 100):
                firstElement = goalSuccessTrack[goal].popleft()
                goalSuccessCount[goal] -= firstElement
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
                time.sleep(0.05)
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                stepCount += 1
                episodeSteps += 1
                #print episodeSteps
                nextState = env.getStackedState()
                                                
                # Update data for visualization
                externalRewardMonitor += externalRewards

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                goalSuccessTrack[goal].append(0)
                break
            elif env.goalReached(goal):
                goalSuccessTrack[goal].append(1)
                goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                
                #lastGoal = goal
                # get key
                #if goal == 2:
                #    break
            else:
                goalSuccessTrack[goal].append(0)
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

            
            goal = 3
            if (len(goalSuccessTrack[goal]) > 100):
                firstElement = goalSuccessTrack[goal].popleft()
                goalSuccessCount[goal] -= firstElement
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
                time.sleep(0.05)
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                stepCount += 1
                episodeSteps += 1
                #print episodeSteps
                nextState = env.getStackedState()
                                                
                # Update data for visualization
                externalRewardMonitor += externalRewards

            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                goalSuccessTrack[goal].append(0)
                break
            elif env.goalReached(goal):
                goalSuccessTrack[goal].append(1)
                goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                time.sleep(60)
                #lastGoal = goal
                # get key
                if goal == 3:
                    for i in range(15):
                        env.act(3)
                    for i in range(15):
                        env.act(0)
                    break
            else:
                goalSuccessTrack[goal].append(0)
                break
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()
        
        episodeCount += 1


if __name__ == "__main__":
    main()
