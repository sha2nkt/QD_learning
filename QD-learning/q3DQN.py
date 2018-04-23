#!/usr/bin/env python
import tensorflow as tf, numpy as np, gym, sys, copy, argparse
from collections import deque
import random
import os,shutil
from gym.wrappers import Monitor
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, nS, nA):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		
		n_hidden1 = 30
		n_hidden2 = 30 

		X = tf.placeholder(tf.float64, [None, nS], name="input")
		Y_ = tf.placeholder(tf.float64, [None, nA], name="target")

		w1 = tf.Variable(np.random.randn(nS, n_hidden1), dtype = tf.float64, name="w1")
		w2 = tf.Variable(np.random.randn(n_hidden1, n_hidden2), dtype=tf.float64, name="w2")
		w3 = tf.Variable(np.random.randn(n_hidden2, nA), dtype=tf.float64, name="w3")

		b1 = tf.Variable(np.random.randn(n_hidden1))
		b2 = tf.Variable(np.random.randn(n_hidden2))
		b3 = tf.Variable(np.random.randn(nA))

		# check1 = tf.matmul(X, w1)

		prehidden_1 = tf.add(tf.matmul(X, w1), b1)   ## [None, nhidden1]
		posthidden_1 = tf.nn.relu(prehidden_1)

		prehidden_2 = tf.add(tf.matmul(posthidden_1, w2), b2)
		posthidden_2 = tf.nn.relu(prehidden_2)

		Y = tf.add(tf.matmul(posthidden_2, w3), b3)

		loss = tf.losses.mean_squared_error(Y_, Y)

		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		update = optimizer.minimize(loss)

		self.update = update 
		self.X = X
		self.Y_ = Y_
		self.Y = Y 
		# self.b1 = b1 
		# self.check1 = check1 
		# self.prehidden_1 = prehidden_1

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		pass

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		pass

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		self.replay = deque(maxlen=memory_size)
		self.burn_in = burn_in

		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		return random.sample(self.replay, batch_size)

	def append(self, transition):
		# Appends transition to the memory. 
		self.replay.append(transition)	

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.env = gym.make(environment_name)
		self.nS = len(self.env.observation_space.sample())
		self.nA = self.env.action_space.n 
		if (environment_name=='MountainCar-v0'):
			self.gamma = 1.0
		else:
			self.gamma = 0.99
		self.epsilon_min = 0.05
		self.epsilon_max = 0.5
		self.epsilon = self.epsilon_max
		self.maxIterations = 1000000
		self.net = QNetwork(nS=self.nS, nA=self.nA)
		self.diff_net = QNetwork(nS=self.nS, nA=self.nA)
		self.memory = Replay_Memory() 
		self.numIter = 0
		self.numEpisodes = 5000
		self.filepath = './qd_1/'
		shutil.rmtree(self.filepath)  # will delete tb directory and all its contents.
		print('********Tensorboard Log deleted*******')



	def d_greedy_policy(self, state):
		# Creating epsilon greedy probabilities to sample from.
		state = np.reshape(state, (1, self.nS))		
		D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: state})
		return np.argmax(D_val[0])

	def d_random_policy(self, state):
		# Pure exploratory policy, with stress on increasing distance from start state
		if np.random.rand() <= 0.5:
			return self.env.action_space.sample()
		else:
			state = np.reshape(state, (1,self.nS))
			D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: state})
			return 	np.argmax(D_val[0])

	def q_greedy_policy(self, state): # change this to include epsilon

		state = np.reshape(state,(1,self.nS))
		Q_val = self.sess.run(self.net.Y, feed_dict={self.net.X: state})
		return np.argmax(Q_val[0])




	def train(self, batch_size=32):
		minibatch = self.memory.sample_batch(batch_size)
		batchState = np.zeros([batch_size, self.nS])
		batchTarget = np.zeros([batch_size, self.nA])
		batchTarget_diff = np.zeros([batch_size, self.nA])
		j = 0 
		for state, action, reward, nextState, isTerminal, diff in minibatch:

			target = reward
			target_diff = diff

			if not isTerminal:
				nextState = np.reshape(nextState,(1,self.nS))
				Q_val = self.sess.run(self.net.Y, feed_dict={self.net.X: nextState})
				D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: nextState})
				target = reward + self.gamma*np.amax(Q_val[0])
				target_diff = diff + self.gamma*np.amax(D_val[0])

			state = np.reshape(state,(1,self.nS))
			batchState[j] = state

			Y_ = self.sess.run(self.net.Y, feed_dict={self.net.X: state})
			Y_diff = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: state})
			Y_[0][action] = target
			Y_diff[0][action] = target_diff

			batchTarget[j] = Y_[0]
			batchTarget_diff[j] = Y_diff[0]

			j += 1
		self.sess.run(self.net.update, feed_dict={self.net.X: batchState, self.net.Y_: batchTarget})
		self.sess.run(self.diff_net.update, feed_dict={self.diff_net.X: batchState, self.diff_net.Y_: batchTarget_diff})

		self.numIter = self.numIter + 1
		# if (self.epsilon > self.epsilon_min):
		# 	# print self.epsilon
		# 	self.epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min)*self.numIter/self.maxIterations

	def act(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		init = tf.global_variables_initializer()

		self.sess = tf.Session(config=config)

		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(self.filepath, self.sess.graph)

		self.sess.run(init)

		saver = tf.train.Saver(max_to_keep = 20)
		fileNum = 1

		self.burn_in_memory()

		for i in range(0, self.numEpisodes):
			self.diffmax = 0 # to regularize diff between 0 and 1
			state = self.env.reset()
			start_state = state
			isTerminal = False 
			step = 0
			if(i%1000==0):
				print('Episode Num:', i)

			while (not isTerminal):

				self.train(batch_size = 32)
				# decide action
				# with prob=epsilon, take D random
				if np.random.rand() <= self.epsilon:
					action = self.d_random_policy(state)
					nextState, reward, isTerminal, _ = self.env.step(action)
				else:
					action = self.q_greedy_policy(state)
					nextState, reward, isTerminal, _ = self.env.step(action)
					if reward is not 0:
						# keep the q-greedy action
						pass
					else:
						if np.random.rand() <= self.epsilon:
							action = self.d_greedy_policy(state)
							nextState, reward, isTerminal, _ = self.env.step(action)
						else:
							# keep the q-greedy action
							pass

				diff = np.sum(np.square(nextState - start_state))
				if diff > self.diffmax:
					self.diffmax = diff
				diff = diff/self.diffmax

				step += 1
				transition = (state, action, reward, nextState, isTerminal, diff)

				self.memory.append(transition)

				state = copy.deepcopy(nextState)

			if (i % 100 == 0):
				testAcc = self.test()
				print (testAcc)

				if (i % 100 == 0):
					saver.save(self.sess, self.filepath + 'network_' + str(fileNum).zfill(3) + '.ckpt')
					fileNum += 1
				
				summary = tf.Summary() 
				summary.ParseFromString(self.sess.run(summary_op))
				summary.value.add(tag='AvgTestReward', simple_value=float(testAcc))
				summary_writer.add_summary(summary, self.numIter)

		# saver.restore(self.sess, './MC/network_' + str(8).zfill(3) + '.ckpt')
		# self.env = Monitor(self.env, directory='./MountainCar2',video_callable=None, write_upon_reset=True)
		# testAcc = self.test()
		# print testAcc


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		# No notions of diff here... we evaluate based on only rewards


		testEpisodes = 20
		CumReward = 0.0
		# CumDiff = 0.0
		# stdDev = np.zeros(100)
		for t in range(0, testEpisodes):
			# self.diffmax = 0
			state = self.env.reset()
			start_state = state
			# self.env.render()
			isTerminal = False
			rewardEpisode = 0.0
			# diffEpisode = 0.0
			while (not isTerminal):

				state = np.reshape(state,(1,self.nS))
				action = self.q_greedy_policy(state)
				nextState, reward, isTerminal, _ = self.env.step(action)
				# diff = np.sum(np.square(nextState - start_state))
				# if diff > self.diffmax:
				# 	self.diffmax = diff
				# diff = diff/self.diffmax
				# diffEpisode = diffEpisode + diff
				rewardEpisode = rewardEpisode + reward
				state = copy.deepcopy(nextState)
				print('Test episode starts')
				print(t)
				print(rewardEpisode)
				print('Test episode ends')

			# stdDev[t] = rewardEpisode
			# CumDiff = CumDiff + diff
			CumReward = CumReward + rewardEpisode  
		# print np.std(stdDev)
		# print np.mean(stdDev)	

		return (CumReward/testEpisodes)

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		for e in range(0, self.memory.burn_in):

			self.diffmax = 0
			state = self.env.reset()
			start_state = state
			isTerminal = False

			while (not isTerminal):
				action = self.env.action_space.sample()
				nextState, reward, isTerminal, _ = self.env.step(action)
				diff = np.sum(np.square(nextState - start_state))
				if diff > self.diffmax:
					self.diffmax = diff
				diff = diff/self.diffmax

				transition = (state, action, reward, nextState, isTerminal, diff)
				self.memory.append(transition)
				state = nextState

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):
	env = 'MountainCar-v0'
	print('##########' + env +'##########')
	agent = DQN_Agent(environment_name=env)
	agent.act()
	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	# gpu_ops = tf.GPUOptions(allow_growth=True)
	# config = tf.ConfigProto(gpu_options=gpu_ops)
	# sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	# keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)
