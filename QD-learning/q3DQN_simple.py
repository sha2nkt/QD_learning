#!/usr/
# bin/env python
import tensorflow as tf, numpy as np, gym, sys, copy, argparse
from collections import deque
import random
import os,shutil
from gym.wrappers import Monitor
import pdb
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class DNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, nS, nA, lr, momentum):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.

		n_hidden1 = 256
		n_hidden2 = 256

		X = tf.placeholder(tf.float64, [None, nS+1], name="input")
		Y_ = tf.placeholder(tf.float64, [None, nA], name="target")

		w1 = tf.Variable(np.random.randn(nS+1, n_hidden1), dtype=tf.float64, name="w1")
		w2 = tf.Variable(np.random.randn(n_hidden1, n_hidden2), dtype=tf.float64, name="w2")
		w3 = tf.Variable(np.random.randn(n_hidden2, nA), dtype=tf.float64, name="w3")

		b1 = tf.Variable(np.random.randn(n_hidden1))
		b2 = tf.Variable(np.random.randn(n_hidden2))
		b3 = tf.Variable(np.random.randn(nA))

		prehidden_1 = tf.add(tf.matmul(X, w1), b1)  ## [None, nhidden1]
		posthidden_1 = tf.nn.relu(prehidden_1)

		prehidden_2 = tf.add(tf.matmul(posthidden_1, w2), b2)
		posthidden_2 = tf.nn.relu(prehidden_2)

		Y = tf.add(tf.matmul(posthidden_2, w3), b3)

		loss = tf.losses.mean_squared_error(Y_, Y)
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
		update = optimizer.minimize(loss)
		self.loss = loss
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

	def load_model_weights(self, weight_file):
		# Helper funciton to load model weights.
		pass
class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=128):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		self.replay = deque(maxlen=memory_size)
		self.burn_in = burn_in

		pass

	def sample_batch(self, batch_size):
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
			self.gamma = 0.99
		else:
			self.gamma = 0.99
		self.epsilon_min = 0.05
		self.epsilon_max = 0.5
		self.epsilon = self.epsilon_max
		self.maxIterations = 1000000
		self.batch_size = 64
		self.learning_rate = 1e-4
		self.momentum = 0.9
		self.diff_net = DNetwork(nS=self.nS, nA=self.nA, lr=self.learning_rate, momentum=self.momentum)
		self.memory = Replay_Memory() 
		self.numIter = 0
		self.numEpisodes = 5000
		self.filepath = './qd_1_mc_4/'
		print('Tensorboard file contents cleared!!!')
		shutil.rmtree(self.filepath, ignore_errors=True)  # will delete tb directory and all its contents.

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.

		for e in range(0, self.memory.burn_in):
			diff = 0
			state = self.env.reset()
			start_state = state
			isTerminal = False
			while (not isTerminal):
				action = self.env.action_space.sample()
				nextState, reward, isTerminal, _ = self.env.step(action)
				cur_diff = np.sum(np.square(nextState[0] - start_state[0]))
				if cur_diff>diff:
					diff = cur_diff
				# # To ensure that the agent doesn't get stuck
				# if nextState is not state:
				# 	pass
				# else:
				# 	diff = 0

				transition = (state, action, reward, nextState, isTerminal, diff, start_state)
				self.memory.append(transition)
				state = nextState

	def d_greedy_policy(self, state, start_state):
		# Creating epsilon greedy probabilities to sample from.
		state = np.reshape(state, (1, self.nS))
		# start_state = np.reshape(start_state, (1,self.nS))
		input = np.hstack((state, start_state[0].reshape(1,-1)))
		input = np.reshape(input, (1, self.nS+1))
		D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: input})
		return np.argmax(D_val[0])

	def d_random_policy(self, state, start_state):
		# Pure exploratory policy, with stress on increasing distance from start state
		if np.random.rand() <= self.epsilon:
			return self.env.action_space.sample()
		else:
			state = np.reshape(state, (1,self.nS))
			# start_state = np.reshape(start_state, (1, self.nS))
			input = np.hstack((state, start_state[0].reshape(1,-1)))
			input = np.reshape(input, (1, self.nS+1))
			D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: input})
			return 	np.argmax(D_val[0])

	def random_policy(self, state, start_state):
		# Pure exploratory policy, with stress on increasing distance from start state
		return self.env.action_space.sample()

	def epsilon_d_greedy_policy(self, state, start_state):
		# Creating epsilon greedy probabilities to sample from.
		# state = np.reshape(state, (1, self.nS))
		# start_state = np.reshape(start_state, (1,self.nS))
		if np.random.rand() <= self.epsilon_min:
			return self.env.action_space.sample()
		else:
			state = np.reshape(state, (1,self.nS))
			# start_state = np.reshape(start_state, (1, self.nS))
			input = np.hstack((state, start_state[0].reshape(1,-1)))
			input = np.reshape(input, (1, self.nS+1))
			D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: input})
			return np.argmax(D_val[0])

	def train(self, batch_size):
		minibatch = self.memory.sample_batch(batch_size)
		batchInput = np.zeros([batch_size, self.nS+1])
		batchState = np.zeros([batch_size, self.nS])

		batchTarget = np.zeros([batch_size, self.nA])
		batchTarget_diff = np.zeros([batch_size, self.nA])
		for j, (state, action, reward, nextState, isTerminal, diff, start_state) in enumerate(minibatch):

			target_diff = diff

			if not isTerminal:
				nextState = np.reshape(nextState,(1,self.nS))
				# start_state = np.reshape(start_state, (1, self.nS))
				next_input = np.hstack((nextState, start_state[0].reshape(1,-1)))
				next_input = np.reshape(next_input, (1, self.nS+1))
				D_val = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: next_input})
				target_diff = diff + self.gamma*np.amax(D_val[0])

			state = np.reshape(state,(1,self.nS))
			# start_state = np.reshape(start_state, (1, self.nS))
			input = np.hstack((state, start_state[0].reshape(1,-1)))
			input = np.reshape(input, (1, self.nS+1))
			batchInput[j] = input

			Y_diff = self.sess.run(self.diff_net.Y, feed_dict={self.diff_net.X: input})
			Y_diff[0][action] = target_diff

			batchTarget_diff[j] = Y_diff[0]

		_, loss = self.sess.run([self.diff_net.update, self.diff_net.loss], feed_dict={self.diff_net.X: batchInput, self.diff_net.Y_: batchTarget_diff})

		self.numIter = self.numIter + 1

		# Epsilon annealing

		if (self.epsilon > self.epsilon_min):
			self.epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min)*self.numIter/self.maxIterations

		return loss

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

		# summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(self.filepath, self.sess.graph)
		summary = tf.Summary()
		self.sess.run(init)

		saver = tf.train.Saver(max_to_keep = 20)
		fileNum = 1

		self.burn_in_memory()

		for e in range(0, self.numEpisodes):
			diff = 0
			state = self.env.reset()
			start_state = state
			isTerminal = False 
			step = 0
			if(e%1000==0):
				print('Episode Num:', e)

			while (not isTerminal):

				loss = self.train(batch_size = self.batch_size)
				# decide action
				# with prob=epsilon, take D random
				if np.random.rand() <= self.epsilon:
					action = self.random_policy(state, start_state)
					nextState, reward, isTerminal, _ = self.env.step(action)
				else:
					action = self.epsilon_d_greedy_policy(state, start_state)
					nextState, reward, isTerminal, _ = self.env.step(action)
					# if reward is not 0:
					# 	# keep the q-greedy action
					# 	pass
					# else:
					# 	if np.random.rand() <= self.epsilon:
					# 		action = self.d_greedy_policy(state)
					# 		nextState, reward, isTerminal, _ = self.env.step(action)
					# 	else:
					# 		# keep the q-greedy action
					# 		pass
				cur_diff = np.sum(np.square(nextState[0] - start_state[0]))
				if cur_diff>diff:
					diff = cur_diff
				# if nextState is not state:
				# 	pass
				# else:
				# 	diff = 0

				step += 1
				transition = (state, action, reward, nextState, isTerminal, diff, start_state)

				self.memory.append(transition)

				state = copy.deepcopy(nextState)

				# summary.value.add(tag='Training Loss', simple_value=float(loss))

				if (e % 100 == 0 and e is not 0):
					testDiff = self.test()
					print ('Average Test Diff:', testDiff)

					if (e % 100 == 0):
						saver.save(self.sess, self.filepath + 'network_' + str(fileNum).zfill(3) + '.ckpt')
						fileNum += 1


					# summary.ParseFromString(self.sess.run(summary_op)) # merge all tf.summaries
					summary.value.add(tag='AvgTestDiff', simple_value=float(testDiff))

				summary_writer.add_summary(summary, self.numIter)

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		# No notions of diff here... we evaluate based on only rewards


		testEpisodes = 20
		CumDiff = 0.0
		# stdDev = np.zeros(100)
		for e in range(0, testEpisodes):
			# self.diffmax = 0
			diff = 0
			state = self.env.reset()
			start_state = state
			isTerminal = False
			diffEpisode = 0.0
			while (not isTerminal):


				action = self.d_greedy_policy(state, start_state)
				nextState, reward, isTerminal, _ = self.env.step(action)
				cur_diff = np.sum(np.square(nextState[0] - start_state[0]))
				if cur_diff>diff:
					diff = cur_diff
				# if nextState is not state:
				# 	pass
				# else:
				# 	diff = 0

				diffEpisode = diffEpisode + diff
				state = copy.deepcopy(nextState)
				# print('******Test episode starts*******')
				# print('Episode #', t)
				# print('Episode Diff', diffEpisode)
				# print('*******Test episode ends*******')

			# stdDev[t] = rewardEpisode
			# CumDiff = CumDiff + diff
			CumDiff = CumDiff + diffEpisode
		# print np.std(stdDev)
		# print np.mean(stdDev)	

		return (CumDiff/testEpisodes)



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
