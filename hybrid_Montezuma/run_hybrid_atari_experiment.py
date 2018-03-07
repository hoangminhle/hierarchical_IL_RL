# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import os
os.environ['PYTHONHASHSEED'] = '0'

from hyperparameters import *

from environment_atari import ALEEnvironment
from hybrid_rl_il_agent_atari import Agent
from hybrid_model_atari import Hdqn

import argparse
import sys
from collections import namedtuple, deque
#from simple_net import Net
from meta_net_il import MetaNN
from PIL import Image
from tensorboard import TensorboardVisualizer
from os import path
from time import sleep
import pickle

#####

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    # Initilization for tensor board
    visualizer = TensorboardVisualizer()
    logdir = path.join(recordFolder+'/') ## subject to change
    visualizer.initialize(logdir,None)

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    #actionMap = [1, 2, 3, 4, 5, 11, 12] # testing: taking out no np action to see what happens
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    goalExplain = ['lower right ladder',  'key', 'lower right ladder', 'right door']
    
    subgoal_success_tracker = [[] for i in range(4)]
    subgoal_trailing_performance = [0,0,0,0]
    random_experience = [deque(), deque(), deque(), deque()]
    kickoff_lowlevel_training = [False, False, False, False]

    #goalSuccessCount = [0, 0, 0, 0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=False)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=False)
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
    annealComplete = False
    saveExternalRewardScreen = True
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn(GPU)
    #hdqn.loadWeight(0)
    hdqn1 = Hdqn(GPU)
    #hdqn1.loadWeight(1)
    hdqn2 = Hdqn(GPU)
    #hdqn2.loadWeight(2)
    hdqn3 = Hdqn(GPU)
    #hdqn3.loadWeight(3)
    hdqn_list = [hdqn, hdqn1, hdqn2, hdqn3]
    for i in range(4):
        if i not in goal_to_train:
            hdqn_list[i].loadWeight(i) # load the pre-trained weights for subgoals that are not learned?
            kickoff_lowlevel_training[i] = True # switch this off
    
    ## Initialize agents and metacontrollers
    agent = Agent(hdqn, range(nb_Action), range(4), defaultNSample=BATCH, defaultRandomPlaySteps=1000, controllerMemCap=EXP_MEMORY, explorationSteps=50000, trainFreq = TRAIN_FREQ, hard_update = 1000)
    agent1 = Agent(hdqn1, range(nb_Action), range(4), defaultNSample=BATCH, defaultRandomPlaySteps=20000, controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq = TRAIN_FREQ, hard_update = HARD_UPDATE_FREQUENCY)
    agent2 = Agent(hdqn2, range(nb_Action), range(4), defaultNSample=BATCH, defaultRandomPlaySteps=20000, controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq = TRAIN_FREQ, hard_update=HARD_UPDATE_FREQUENCY)
    agent3 = Agent(hdqn3, range(nb_Action), range(4), defaultNSample=BATCH, defaultRandomPlaySteps=20000, controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq = TRAIN_FREQ, hard_update = HARD_UPDATE_FREQUENCY)
    agent_list = [agent, agent1, agent2, agent3]
    metacontroller = MetaNN()

    for i in range(4):
        #if i in goal_to_train:
        agent_list[i].compile()
        if i not in goal_to_train:
            agent_list[i].randomPlay = False
            agent_list[i].controllerEpsilon = 0.0

    externalRewardMonitor = 0
    totalIntrinsicReward = 0
    subgoalTotalIntrinsic = [0, 0, 0, 0]
    option_learned = [False, False, False, False]
    training_completed = False
    for i in range(4):
        if i not in goal_to_train:
            option_learned[i] = True
    episodeCount = 0
    stepCount = 0

    option_t = [0, 0, 0, 0]
    option_training_counter = [0 , 0 ,0 , 0]
    meta_training_counter = 0

    #for episode in range(80000):
    record = []
    meta_count = 0
    wrong_meta_pred = 0
    while episodeCount < EPISODE_LIMIT and stepCount < STEPS_LIMIT and (not training_completed):
        print("\n\n### EPISODE "  + str(episodeCount) + "###")
        print("\n\n### STEPS "  + str(stepCount) + "###")
        #print("Current controller epsilon for goal is", agent.controllerEpsilon[3])
        for subgoal in range(4):
            print "Current epsilon for subgoal ", str(subgoal), " is:", agent_list[subgoal].controllerEpsilon
        print
        for subgoal in range(4):
            print "Number of samples for subgoal ", str(subgoal), " is:", option_t[subgoal]
        print
        # Restart the game
        #sleep(2)
        env.restart()

        decisionState = env.getStackedState()

        meta_labels = []
        wrong_option = False

        episodeSteps = 0

        goal = metacontroller.sample(metacontroller.predict(decisionState))
        #goal = 0 # ground truth
        true_goal = 0
        expert_goal = np.zeros((1,nb_Option))
        expert_goal[0, true_goal] = 1.0
        meta_labels.append((decisionState, expert_goal)) # append, but do not collect yet 
        
        meta_count += 1
        
        if goal!= true_goal:
            wrong_option = True
            print "Terminate because picking wrong option at goal", true_goal
            wrong_meta_pred += 1
            print "Number of wrong meta choices: ", wrong_meta_pred
            if wrong_meta_pred % 100 == 0:
                metacontroller.reset()
                print "Resetting the meta controller"
                #sleep(2)


        loss_list = []
        avgQ_list = []
        tdError_list = []

        # set goalNum to hardcoded subgoal
        while not env.isTerminal() and episodeSteps <= maxStepsPerEpisode and (not wrong_option):
            #totalIntrinsicReward = 0
            totalExternalRewards = 0 # NOT SURE IF IT SHOULD BE CLEARED HERE!
            
            #stateLastGoal = env.getStackedState()
            # nextState = stateLastGoal
            
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode and(not wrong_option):
                state = env.getStackedState()
                #action = agent_list[goal].selectMove(state, goal)
                action = agent_list[goal].selectMove(state)
                externalRewards = env.act(actionMap[action])
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                #stepCount += 1
                episodeSteps += 1
                nextState = env.getStackedState()
                # only assign intrinsic reward if the goal is reached and it has not been reached previously
                intrinsicRewards = agent_list[goal].criticize(env.goalReached(goal), actionMap[action], env.isTerminal(), 0, args.use_sparse_reward)
                # Store transition and update network params
                if agent_list[goal].randomPlay:
                    exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                    random_experience[goal].append(exp)
                    if len(random_experience[goal]) > 20000:
                        random_experience[goal].popleft()
                    #print "Length of random experience bank is ", len(random_experience[goal])
                else:
                    if not kickoff_lowlevel_training[goal]:
                        for exp in random_experience[goal]:
                            agent_list[goal].store(exp, meta=False)
                            option_t[goal] += 1
                            option_training_counter[goal] += 1
                        print "Finally, the number of stuff in random_experience is", len(random_experience[goal])
                        print "The number of item in experience memory so far is:", len(agent_list[goal].memory)
                        random_experience[goal].clear()
                        assert len(random_experience[goal]) == 0
                        kickoff_lowlevel_training[goal] = True
                        print "This should really be one time thing"
                        print " number of option_t is ", option_t[goal]
                        print 
                        #sleep(10)
                    else:
                        if not option_learned[goal]:
                            exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                            agent_list[goal].store(exp, meta=False)
                            option_t[goal] += 1
                            option_training_counter[goal] += 1
                
                # Do not update the network during random play
                if (option_t[goal] >= agent_list[goal].defaultRandomPlaySteps) and (not agent_list[goal].randomPlay):
                    if (option_t[goal] == agent_list[goal].defaultRandomPlaySteps):
                        print('start training (random walk ends) for subgoal '+str(goal))

                    if (option_t[goal] % agent_list[goal].trainFreq == 0 and option_training_counter[goal]>0 and (not option_learned[goal])):
                        loss, avgQ, avgTDError = agent_list[goal].update(option_t[goal], meta=False)
                        loss_list.append(loss)
                        avgQ_list.append(avgQ)
                        tdError_list.append(avgTDError)
                        option_training_counter[goal] = 0
                                                    
                totalExternalRewards += externalRewards
                totalIntrinsicReward += intrinsicRewards
                subgoalTotalIntrinsic[goal] += intrinsicRewards
                
                # Update data for visualization
                externalRewardMonitor += externalRewards
            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                subgoal_success_tracker[goal].append(0)
                break
            elif env.goalReached(goal):
                subgoal_success_tracker[goal].append(1)
                #goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                print
                if agent_list[goal].randomPlay:
                    agent_list[goal].randomPlay = False
                    #option_t[goal] = 0 ## Reset option counter 
                episodeSteps = 0 ## reset episode steps to give new goal all 500 steps
                
                decisionState = env.getStackedState()
                
                goal = metacontroller.sample(metacontroller.predict(decisionState))
                #print "Next predicted goal is:", goal
                print('Next predicted subgoal is: ' + goalExplain[goal])
                
                #goal = goal+1 ## alternatively, try setting goal to the ground truth goal
                true_goal = true_goal + 1

                #goal = goal +1
                if true_goal <nb_Option:
                    meta_count +=1
                    expert_goal = np.zeros((1,nb_Option))
                    expert_goal[0, true_goal] = 1.0
                    meta_labels.append((decisionState, expert_goal)) # append, but do not collect yet 

                # get key
                if true_goal == nb_Option:
                    break
                
                if goal!= true_goal:
                    wrong_option = True
                    print "Terminate because picking wrong option at goal", true_goal
                    wrong_meta_pred += 1
                    print "Number of wrong meta choices: ", wrong_meta_pred
                    if wrong_meta_pred % 100 == 0:
                        metacontroller.reset() ## Resetting the meta-controller and retrain. This is fine because we're doing DAgger at the top level
                    break

            else:
                if not wrong_option:
                    subgoal_success_tracker[goal].append(0)
                break
                if not env.isGameOver():
                    env.beginNextLife()
        
        stepCount = sum(option_t)
        if stepCount > 10000: ## Start plotting after certain number of steps
            for subgoal in range(nb_Option):
                visualizer.add_entry(option_t[subgoal], "trailing success ratio for goal "+str(subgoal), subgoal_trailing_performance[subgoal]) 
            visualizer.add_entry(stepCount, "average Q values", np.mean(avgQ_list))
            visualizer.add_entry(stepCount, "training loss", np.mean(loss_list))
            visualizer.add_entry(stepCount, "average TD error", np.mean(tdError_list))
            visualizer.add_entry(stepCount, "episodic intrinsic reward", float(totalIntrinsicReward))
            visualizer.add_entry(stepCount, "total intrinsic reward second subgoal", float(subgoalTotalIntrinsic[2]))
            visualizer.add_entry(stepCount, "total intrinsic reward third subgoal", float(subgoalTotalIntrinsic[3]))
            visualizer.add_entry(stepCount, "total environmental reward", float(externalRewardMonitor))

        episodeCount += 1

        item = meta_labels[-1]
        metacontroller.collect(item[0], item[1]) ## Aggregate training data for the meta controller
        meta_training_counter += 1
        
        if metacontroller.check_training_clock() and (meta_training_counter >= 20):
            print "training metacontroller"
            meta_loss = metacontroller.train()
            meta_training_counter = 0 # reset counter
        
        print
        for subgoal in range(nb_Option):
            if len(subgoal_success_tracker[subgoal]) > 100:
                subgoal_trailing_performance[subgoal] = sum(subgoal_success_tracker[subgoal][-100:])/ 100.0
                if subgoal_trailing_performance[subgoal] > STOP_TRAINING_THRESHOLD:
                    if not option_learned[subgoal]:
                        option_learned[subgoal] = True
                        hdqn_list[subgoal].saveWeight(subgoal)
                        agent_list[subgoal].clear_memory(subgoal)
                        hdqn_list[subgoal].clear_memory()
                        print "Training completed after for subgoal", subgoal, "Model saved"
                        if subgoal == (nb_Option-1):
                            training_completed = True ## Stop training, all done
                    else:
                        print "Subgoal ", subgoal, " should no longer be in training"
                elif subgoal_trailing_performance[subgoal] < STOP_TRAINING_THRESHOLD and option_learned[subgoal]:
                    print "For some reason, the performance of subgoal ", subgoal, " dropped below the threshold again"
            else:
                subgoal_trailing_performance[subgoal] = 0.0
            print "Trailing success ratio for "+str(subgoal)+" is:", subgoal_trailing_performance[subgoal]

        record.append( (episodeCount, stepCount, option_t[0], subgoal_trailing_performance[0], option_t[1], subgoal_trailing_performance[1], option_t[2], subgoal_trailing_performance[2], option_t[3], subgoal_trailing_performance[3], meta_count, true_goal, metacontroller.meta_ind))
        if episodeCount % 100 == 0 or training_completed:
            with open(recordFolder+"/"+recordFileName+".pkl", "wb") as fp:
                pickle.dump(record, fp)
        
        
        if (not annealComplete):
            # Annealing 
            agent.annealMetaEpsilon(stepCount)
            for subgoal in range(4):
                agent_list[subgoal].annealControllerEpsilon(option_t[subgoal], option_learned[subgoal])
    if not option_learned:
    	print "Training terminated after ", stepCount, "steps taken. Option was not learned"

if __name__ == "__main__":
    main()
