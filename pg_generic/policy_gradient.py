# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    policy_gradient.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:35:16 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/28 16:33:33 by jcruz-y-         ###   ########.fr        #
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

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.9,
        steps=100,
        load_path=None,
        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.steps = steps
        self.neurons_layer_1 = 800
        self.neurons_layer_2 = 800

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        # Initialize empty arrays to store states (observs, action, rewards)
        self.batch_observations, self.batch_actions, self.game_rewards, self.batch_rewards = [], [], [], []
        
        # Build network
        self.build_network()

        # Array to store cost history
        self.cost_history = []

        # Initialize session, point to graph, don't run nodes
        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        from time import gmtime, strftime
        s = strftime("%a_%d_%b_%Y_%H:%M", gmtime())
        self.writer = tf.summary.FileWriter('logs/%s/' % s,self.sess.graph)
        # Initialize nodes with global variables
        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            saver = tf.train.import_meta_graph(load_path + ".meta")
            self.saver.restore(self.sess, self.load_path)
            print("model restored")
    
    # 0. Build Network
    def build_network(self):
        # Create placeholders for inputs : self.X and outputs : self.Y and discounted rewards normalized = self.discoun...
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_batch_rewards_norm = tf.placeholder(tf.float32, [None, ], name="rewards")

        # Initialize parameters
        units_layer_1 = self.neurons_layer_1
        units_layer_2 = self.neurons_layer_2 
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
        labels = tf.transpose(self.Y, name='actions_one_hot_v_t')
        self.outputs_softmax = tf.nn.softmax(logits, name='softy')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_batch_rewards_norm)  # reward guided loss
            print("\n\nSELF GAME REWARDS", self.game_rewards, "\n\n")
            tf.summary.scalar('loss', loss)
            self.summaries = tf.summary.merge_all()

        with tf.name_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, global_step=self.global_step)

       # with tf.name_scope('rewards'):
         #   reward_mean = tf.metrics.mean(self.game_rewards)
         #   tf.summary.scalar('reward_mean', reward_mean)
            #get_variable(name="rew_mean", np.mean(self.game_rewards)

    # 1. Choose action based on observation (state)
    def choose_action(self, observation):
        """
            Choose action based on observation
            Arguments:
                observation: array of state, has shape (num_features)
            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        #observation = preprocess(observation)
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
        # Store batch observations
        self.batch_observations.append(s)

        # Store game and batch rewards
        self.game_rewards.append(r)
        self.batch_rewards.append(r)

        # Store batch actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]),
        # array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.batch_actions.append(action)


    # 3. Learn 
    def learn(self):
        # Discount and normalize episode reward
        discounted_batch_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode (batch)
        _, summaries, global_step = self.sess.run([self.train_op, self.summaries, self.global_step], feed_dict={
             self.X: np.vstack(self.batch_observations).T,
             self.Y: np.vstack(np.array(self.batch_actions)).T,
             self.discounted_batch_rewards_norm: discounted_batch_rewards_norm,
        })
        self.writer.add_summary(summaries, global_step)
        self.writer.flush()

        # Reset the episode data
        self.batch_observations, self.batch_actions, self.batch_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
           # print("Model saved in file: %s" % save_path)

        return discounted_batch_rewards_norm

    # 3.1 Discount and normalize rewards
    def discount_and_norm_rewards(self):
        discounted_batch_rewards = np.zeros_like(self.batch_rewards, dtype=float)
        cumulative = 0
        for t in reversed(range(len(self.batch_rewards))):
            if t % self.steps == 0:
                cumulative = 0
            cumulative = cumulative * self.gamma + self.batch_rewards[t]
            discounted_batch_rewards[t] = cumulative

        discounted_batch_rewards -= np.mean(discounted_batch_rewards)
        discounted_batch_rewards /= np.std(discounted_batch_rewards)
       # print("Batch Rewards: \n", self.batch_rewards)
        #print("Discounted batch rewards: \n", discounted_batch_rewards)
        return discounted_batch_rewards


    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
