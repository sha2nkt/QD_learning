import tensorflow as tf, numpy as np, gym, sys, copy, argparse 
from gridworldv2 import gameEnv 
import tensorflow.contrib.slim as slim 

class HeroNetwork():


	def __init__(self):

		self.X = tf.placeholder(tf.float32, shape=[None, 81*5])
		## Obstacle poses at t-1, Obstacle pose at t 

		self.Y = tf.placeholder(tf.float32, shape=[None, 81])

		## 3x3 kernels, 2 input channels, 8 output channels  

		self.w1 = tf.Variable(np.random.randn(81*5, 200), dtype=tf.float32)
		self.b1 = tf.Variable(np.random.randn(200), dtype=tf.float32)
		# ## 5*5*16 = 400 

		self.fc1 = tf.add(tf.matmul(self.X, self.w1), self.b1)

		self.w2 = tf.Variable(np.random.randn(200, 200), dtype=tf.float32)
		self.b2 = tf.Variable(np.random.randn(200), dtype=tf.float32)
		self.fc2 = tf.add(tf.matmul(self.fc1, self.w2), self.b2)

		self.w3 = tf.Variable(np.random.randn(200, 81), dtype=tf.float32)
		self.b3 = tf.Variable(np.random.randn(81), dtype=tf.float32)
		self.fc3 = tf.add(tf.matmul(self.fc2, self.w3), self.b3)
		# ## this is my logits now 

		self.prob = tf.nn.sigmoid(self.fc3)

		# # loss = tf.losses.mean_squared_error(Y, output)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.fc3))

		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		self.update = self.optimizer.minimize(self.loss)


class Agent():

	def __init__(self):

		self.env = gameEnv(partial = False, size = 9)
		self.max_epLength = 100 

	def train(self):

		net = HeroNetwork()
		init = tf.global_variables_initializer()

		sess = tf.Session()
		summary_writer = tf.summary.FileWriter('./agentPred', sess.graph)
		sess.run(init)

		num_episodes = 10000  
		nS = 81 

		for n in range(0, num_episodes):

			avgLoss = 0.0
			states, actions, nextStates = self.generate_training_data()
			T = len(states)
			# T = 15 

			X = np.zeros((T, 81*5))
			Y = np.zeros((T, 81))

			for i in range(1, T):

				a = actions[i]
				X[i,a*nS:a*nS+nS] = np.reshape(states[i][:,:,2], [81]) ## agent 

				Y[i,:] = np.reshape(nextStates[i][:,:,2], [81])
				## column-wise concatenation 
			batch_size = 4
			num_steps = int(T/batch_size)

			for step in range(num_steps):
				j = (step*batch_size)
				sess.run(net.update, feed_dict={net.X: X[j:(j+batch_size),:],
					net.Y: Y[j:(j+batch_size),:]})				
				loss = sess.run(net.loss, feed_dict={net.X: X[j:(j+batch_size),:],
					net.Y: Y[j:(j+batch_size),:]})
				avgLoss += loss 

			avgLoss = (avgLoss/num_steps)
			print avgLoss

			summary = tf.Summary()
			summary.value.add(tag='Average Loss', simple_value=float(avgLoss))
			summary_writer.add_summary(summary, n)

		return sess 

	def save_model_weights(self, sess):

		saver = tf.train.Saver()
		saver.save(sess, './agentDynamics.ckpt')

	def predictNextState(self, sess, net, curr_state, action):

		nS = 81 
		X = np.zeros([1,81*5])
		X[0,action*nS: action*nS + nS] = np.reshape(curr_state[:,:,2], [81]) 

		pS = sess.run(net.prob, feed_dict={net.X:X})
		## predicted state = (1, 81)

		next_state = np.reshape(pS, [9,9])
		return next_state       ## [9,9]


	def generate_training_data(self):

		states = []
		actions = []
		nextStates = []

		s = self.env.reset()

		j = 0 

		while (j < self.max_epLength):

			states.append(s)
			a = np.random.randint(0, 5)
			actions.append(a)
			s1, r, d = self.env.step(a)
			nextStates.append(s1)
			s = s1 
			j += 1 

		return states, actions, nextStates

	def test(self, sess, net):

		curr_state = self.env.reset()
		a = np.random.randint(0,5)

		print curr_state[:,:,2]
		print a 
		next_state, r, d = self.env.step(a)

		pred_state = self.predictNextState(sess, net, curr_state, a)

		print pred_state>0.5

		print next_state[:,:,2]

	def load_transition_network(self):

		net = HeroNetwork()
		init = tf.global_variables_initializer()
		sess = tf.Session()

		sess.run(init)

		saver = tf.train.Saver() 

		saver.restore(sess, './agentDynamics.ckpt')

		return sess, net, saver


def main():

	agent = Agent()
	# sess = agent.train()
	# agent.save_model_weights(sess)

	sess, net, saver = agent.load_transition_network()
	agent.test(sess, net)

if __name__ == '__main__':
	main()