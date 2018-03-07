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
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Lambda, Conv2D, Flatten
from keras import optimizers
from keras import initializers

# Default architectures for the lower level controller/actor
defaultEpsilon = 1.0
defaultControllerEpsilon = 1.0
defaultTau = 0.001
#defaultTau = 1.0

defaultAnnealSteps = 200000
defaultEndEpsilon = 0.01
#defaultRandomPlaySteps = 20000

defaultMetaEpsilon = 1
defaultMetaNSamples = 32
#controllerMemCap = 200000
metaMemCap = 50000
maxReward = 1
minReward = -1
#trueSubgoalOrder = [0, 1, 2]
hardUpdateFrequency = 10000 * 4

prioritized_replay_alpha = 0.6
max_timesteps=1000000
prioritized_replay_beta0=0.4
prioritized_replay_eps=1e-6
prioritized_replay_beta_iters = max_timesteps*0.5
beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                               initial_p=prioritized_replay_beta0,
                               final_p=1.0)

#exploration = LinearSchedule(schedule_timesteps=int(0.2 * 1000000),
#                             initial_p=1.0,
#                             final_p=0.01)

class Agent:

    def __init__(self, net, actionSet, goalSet, defaultNSample, defaultRandomPlaySteps, controllerMemCap, explorationSteps, trainFreq, hard_update, metaEpsilon=defaultMetaEpsilon, epsilon=defaultEpsilon,
                 controllerEpsilon=defaultControllerEpsilon, tau=defaultTau):
        self.actionSet = actionSet
        self.controllerEpsilon = controllerEpsilon
        self.goalSet = goalSet
        self.metaEpsilon = metaEpsilon
        self.nSamples = defaultNSample 
        self.metaNSamples = defaultMetaNSamples 
        self.gamma = defaultGamma
        self.targetTau = tau
        self.net = net
        self.memory = PrioritizedReplayBuffer(controllerMemCap, alpha=prioritized_replay_alpha)
        self.metaMemory = PrioritizedReplayBuffer(metaMemCap, alpha=prioritized_replay_alpha)
        self.enable_double_dqn = True
        self.exploration = LinearSchedule(schedule_timesteps = explorationSteps, initial_p = 1.0, final_p = 0.02)
        self.defaultRandomPlaySteps = defaultRandomPlaySteps
        self.trainFreq = trainFreq
        self.randomPlay = True
        self.learning_done = False
        self.hard_update = hard_update
        #self.metaMemory = Memory(metaMemCap)
    """
    def selectMove(self, state, goal):
        goalVec = utils.oneHot(goal)
        if self.controllerEpsilon[goal] < random.random():
            # predict action
            dummyYtrue = np.zeros((1, 8))
            dummyMask = np.zeros((1, 8))
            return np.argmax(self.net.controllerNet.predict([np.reshape(state, (1, 84, 84, 4)), np.asarray([goalVec]), dummyYtrue, dummyMask], verbose=0)[1])
        return random.choice(self.actionSet)
    """
    def selectMove(self, state):
        if not self.learning_done:
            if self.controllerEpsilon < random.random():
                return np.argmax(self.net.controllerNet.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0))
                #return np.argmax(self.net.controllerNet.predict([np.reshape(state, (1, 84, 84, 4)), dummyYtrue, dummyMask], verbose=0)[1])
            return random.choice(self.actionSet)
        else:
            return np.argmax(self.simple_net.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0))

    def setControllerEpsilon(self, epsilonArr):
        self.controllerEpsilon = epsilonArr

    def selectGoal(self, state):
        if self.metaEpsilon < random.random():
            # predict action
            pred = self.net.metaNet.predict([np.reshape(state, (1, 84, 84, 4)), np.zeros((1,3)), np.zeros((1,3))], verbose=0)[1]
            return np.argmax(pred)
        return random.choice(self.goalSet)

    def selectTrueGoal(self, goalNum):
        return trueSubgoalOrder[goalNum]

    def setMetaEpsilon(self, epsilon):
        self.metaEpsilon = epsilon

    def criticize(self, reachGoal, action, die, distanceReward, useSparseReward):
        reward = 0.0
        if reachGoal:
            reward += 1.0
            #reward += 50.0
        if die:
            reward -= 1.0
        if not useSparseReward:
            reward += distanceReward
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    def store(self, experience, meta=False):
        if meta:
            self.metaMemory.add(experience.state, experience.goal, experience.reward, experience.next_state, experience.done)
        else:
            self.memory.add(experience.state, experience.action, experience.reward, experience.next_state, experience.done)
            #self.memory.add(np.abs(experience.reward), experience)

    def compile(self):
        def huber_loss(y_true, y_pred, clip_value):
            # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
            # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
            # for details.
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
        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.net.controllerNet.output
        y_true = Input(name='y_true', shape=(nb_Action,))
        mask = Input(name='mask', shape=(nb_Action,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        ins = [self.net.controllerNet.input] if type(self.net.controllerNet.input) is not list else self.net.controllerNet.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        #combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        #trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        #opt = optimizers.Adam(lr=0.0001, clipnorm = 10)
        trainable_model.compile(optimizer=rmsProp, loss=losses)
        self.trainable_model = trainable_model
        #print self.trainable_model.summary()
        #print self.trainable_model.metrics_names
        #time.sleep(5)

        self.compiled = True

    def _update(self, stepCount):
        batches = self.memory.sample(self.nSamples, beta=beta_schedule.value(stepCount))
        (stateVector, actionVector, rewardVector, nextStateVector, doneVector, importanceVector, idxVector) = batches
        
        stateVector = np.asarray(stateVector)
        nextStateVector = np.asarray(nextStateVector)
        
        q_values = self.net.controllerNet.predict(stateVector)
        assert q_values.shape == (self.nSamples, nb_Action)
        if self.enable_double_dqn:
            actions = np.argmax(q_values, axis = 1)
            assert actions.shape == (self.nSamples,)

            target_q_values = self.net.targetControllerNet.predict(nextStateVector)
            assert target_q_values.shape == (self.nSamples, nb_Action)
            q_batch = target_q_values[range(self.nSamples), actions]
            assert q_batch.shape == (self.nSamples,)
        else:
            target_q_values = self.net.targetControllerNet.predict(nextStateVector)
            q_batch = np.max(target_q_values, axis=1)
            assert q_batch.shape == (self.nSamples,)

        targets = np.zeros((self.nSamples, nb_Action))
        dummy_targets = np.zeros((self.nSamples,))
        masks = np.zeros((self.nSamples, nb_Action))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        terminalBatch = np.array([1-float(done) for done in doneVector])
        #print "terminal batch"
        #print terminalBatch
        #time.sleep(1)
        assert terminalBatch.shape == (self.nSamples,)
        discounted_reward_batch *= terminalBatch
        reward_batch = np.array(rewardVector)
        action_batch = np.array(actionVector)
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
            #td_error = target[action] - q_values[idx, action]
            #self.memory.update_priorities([idxVector[idx]], [np.abs(td_error) + prioritized_replay_eps])
        #td_errors = np.asarray([targets[i][action_batch[i]]  - q_values[i][action_batch[i]] for i in range(self.nSamples)])
        #td_errors = targets[range(self.nSamples), action_batch] - q_values[range(self.nSamples), action_batch]
        td_errors = targets[range(self.nSamples), action_batch] - q_values[range(self.nSamples), action_batch]
        
        #print np.mean(np.abs(td_errors))
        #time.sleep(0.1)

        new_priorities = np.abs(td_errors) + prioritized_replay_eps
        self.memory.update_priorities(idxVector, new_priorities)
        
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        
        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [stateVector] if type(self.net.controllerNet.input) is not list else stateVector
        if stepCount >= self.defaultRandomPlaySteps:
            loss = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets], sample_weight = [np.array(importanceVector), np.ones(self.nSamples)])
        else:
            loss = [0.0,0.0,0.0]
        
        if stepCount > self.defaultRandomPlaySteps and stepCount % self.hard_update == 0:
            self.net.targetControllerNet.set_weights(self.net.controllerNet.get_weights())
        return loss[1], np.mean(q_values), np.mean(np.abs(td_errors))
        

    def _update_meta(self, stepCount):
        batches = self.metaMemory.sample(self.metaNSamples)
        stateVectors = np.asarray([batch[1].state for batch in batches])
        nextStateVectors = np.asarray([batch[1].next_state for batch in batches])
        
        rewardVectors = self.net.metaNet.predict([stateVectors, np.zeros((self.nSamples,3)), np.zeros((self.nSamples, 3))], verbose=0)[1]
        rewardVectorsCopy = np.copy(rewardVectors)
        rewardVectors = np.zeros((self.metaNSamples, 3))
        nextStateRewardVectors = self.net.targetMetaNet.predict([nextStateVectors, np.zeros((self.nSamples,3)), np.zeros((self.nSamples, 3))], verbose=0)[1]
        maskVector = np.zeros((self.metaNSamples, 3))
        
        for i, batch in enumerate(batches):
            exp = batch[1]
            idx = batch[0]
            maskVector[i, exp.goal] = 1. 
            rewardVectors[i][exp.goal] = exp.reward
            if not exp.done:
                rewardVectors[i][np.argmax(exp.goal)] += self.gamma * max(nextStateRewardVectors[i])
            self.metaMemory.update(idx, np.abs(rewardVectors[i][exp.goal] - rewardVectorsCopy[i][exp.goal]))
        loss = self.net.metaNet.train_on_batch([stateVectors, rewardVectors, maskVector], [np.zeros(self.nSamples), rewardVectors])
        
        #Update target network
        metaWeights = self.net.metaNet.get_weights()
        metaTargetWeights = self.net.targetMetaNet.get_weights()
        for i in range(len(metaWeights)):
            metaTargetWeights[i] = self.targetTau * metaWeights[i] + (1 - self.targetTau) * metaTargetWeights[i]
        self.net.targetMetaNet.set_weights(metaTargetWeights)
        return loss

    def update(self, stepCount, meta=False):
        if meta:
            loss = self._update_meta(stepCount)
        else:
            loss = self._update(stepCount)
        return loss

    def annealMetaEpsilon(self, stepCount):
        self.metaEpsilon = defaultEndEpsilon + max(0, (defaultMetaEpsilon - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - self.defaultRandomPlaySteps)) / defaultAnnealSteps)

    def annealControllerEpsilon(self, stepCount, option_learned):
        #self.controllerEpsilon[goal] = defaultEndEpsilon + max(0, (defaultControllerEpsilon[goal] - defaultEndEpsilon) * \
        #    (defaultAnnealSteps - max(0, 0.25*(stepCount - defaultRandomPlaySteps))) / defaultAnnealSteps)
        if not self.randomPlay:
            if option_learned:
                self.controllerEpsilon = 0.0
            else:
                if stepCount > self.defaultRandomPlaySteps:
                    self.controllerEpsilon = self.exploration.value(stepCount - self.defaultRandomPlaySteps)
                    #self.controllerEpsilon[goal] = exploration.value(stepCount - defaultRandomPlaySteps)
    def clear_memory(self, goal):
        self.learning_done = True ## Set the done learning flag
        del self.trainable_model
        del self.memory
        gpu = self.net.gpu

        del self.net
        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)

        with tf.device('/gpu:'+str(gpu)):
            self.simple_net = Sequential()
            self.simple_net.add(Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84,84,4)))
            self.simple_net.add(Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid'))
            self.simple_net.add(Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid'))
            self.simple_net.add(Flatten())
            self.simple_net.add(Dense(HIDDEN_NODES, activation = 'relu', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            self.simple_net.add(Dense(nb_Action, activation = 'linear', kernel_initializer = initializers.random_normal(stddev=0.01, seed = SEED)))
            self.simple_net.compile(loss = 'mse', optimizer = rmsProp)
            self.simple_net.load_weights(recordFolder+'/policy_subgoal_' + str(goal) + '.h5')
            self.simple_net.reset_states()



