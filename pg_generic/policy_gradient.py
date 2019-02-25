# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    policy_gradient.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:35:16 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/25 11:19:55 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
#import cartpole

def preprocess(state_dict):
	state = np.concatenate((
		np.array(state_dict['ingredients_map']).ravel(),
		np.array(state_dict['slices_map']).ravel(),
		np.array(state_dict['cursor_position']).ravel(),
		[state_dict['slice_mode'],
		state_dict['min_each_ingredient_per_slice'],
		state_dict['max_ingredients_per_slice']],
	))
	return state.astype(np.float).ravel()

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        # Initialize empty arrays to store states (observs, action, rewards)
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        
        # Build network
        self.build_network()

        # Array to store cost history
        self.cost_history = []

        # Initialize session, point to graph, don't run nodes
        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        self.writer = tf.summary.FileWriter("logss/", self.sess.graph)

        # Initialize nodes with global variables
        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)
            print("model restored")
    
    # 0. Build Network
    def build_network(self):
        # Create placeholders for inputs : self.X and outputs : self.Y and discounted rewards normalized = self.discoun...
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="rewards")

        # Initialize parameters
        units_layer_1 = 100
        units_layer_2 = 100
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1, self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
           # A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3, name='tran_log_probs')
        labels = tf.transpose(self.Y, name='tran_vector_size_action_space')
        self.outputs_softmax = tf.nn.softmax(logits, name='softy')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss
            print("\n\nSELF EPISODE REWARDS", self.episode_rewards, "\n\n")
            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, global_step=self.global_step)

        with tf.name_scope('rewards'):
            reward_mean = tf.metrics.mean(self.episode_rewards)
            tf.summary.scalar('reward_mean', reward_mean)
            #get_variable(name="rew_mean", np.mean(self.episode_rewards)
        self.summaries = tf.summary.merge_all()

    # 1. Choose action based on observation (state)
    def choose_action(self, observation):
        """
            Choose action based on observation
            Arguments:
                observation: array of state, has shape (num_features)
            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = preprocess(observation)
        observation = observation[:, np.newaxis]
        
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action
    
    # 2. Store transition
    def store_transition(self, s, a, r):
        """
            Store play memory for training
            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)

        #Store episode(game) rewards
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]),
        # array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)


    # 3. Learn 
    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode (batch)
        _, summaries, global_step = self.sess.run([self.train_op, self.summaries, self.global_step], feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })
        self.writer.add_summary(summaries, global_step)
        self.writer.flush()

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
           # print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    # 3.1 Discount and normalize rewards
    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards, dtype=float)
        print(len(self.episode_rewards))
        print("reward mean from PG: ", np.mean(self.episode_rewards) * 100)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

       # print("MEAN\n", np.mean(discounted_episode_rewards))
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
