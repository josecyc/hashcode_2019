# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    trainer.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:55:13 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/25 15:01:18 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src import game_b
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

# Policy gradient has high variance, seed for reproducability
#env.seed(1)

RENDER_ENV = True
BATCHES = 1000
P_GAMES = 500
STEPS = 100
rewards = []
batch_rewards = []
RENDER_REWARD_MIN = 100
true_max_reward_so_far = 0

R = 6
C = 7
X_DIM = R * C * 2 + 5
#ACTIONS = ["right", "down", "left", "up", "toggle"]
ACTIONS = ["right", "down", "left", "up", "cut_up", "cut_left", "cut_down", "cut_right"]

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

if __name__ == "__main__":

    # Load checkpoint
    load_path = None #"./output/weights/pizza-temp.ckpt"
    save_path = "output/weights/pizza-temp.ckpt"

    PG = PolicyGradient(
            n_x = X_DIM,
            n_y = 8,
            learning_rate=0.01,
            reward_decay=0.95,
            load_path=load_path,
            save_path=save_path
            )

    for batch in range(BATCHES):
        for p_game in range(P_GAMES):
            env = game_b.Game({'max_steps': 100})
            batch_reward = 0
            h = 5			
            l = 1
            pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
            pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            state = env.init(pizza_config)[0]
	    #state[0] #get only first value of tuple
            for step in range(STEPS):
                #if RENDER_ENV: 
                #    env.render()
                # 1. Choose an action based on observation
                action = PG.choose_action(state)

                # 2. Take action in the environment
                state_, reward, done, info = env.step(ACTIONS[action])

                # 3. Store transition for training
                PG.store_transition(preprocess(state), action, reward)
            
                # Save new state
                #state = state_
                if done:
                    batch_rewards.append(sum(PG.game_rewards))
                    batch_rewards_sum = sum(PG.batch_rewards)
                   # print(episode_rewards)
                    #print("batch_rewards_sum", batch_rewards_sum)
                    rewards.append(batch_rewards_sum)
                    #print("partial reward mean", batch_rewards_sum/(p_game + 1))
                    max_reward_so_far = np.amax(batch_rewards)
                    #print("max_reward_so_far", max_reward_so_far)

                    #print("==========================================")
                    #print("p_game: ", p_game)
                    #print("batch: ", batch)
                    #print("Reward: ", episode_rewards_sum)
                    #print("Max Batch reward so far: ", max_reward_so_far)

            #print("game: ", p_game, )
            # 4. Train neural network
            PG.game_rewards = []
        reward_mean = batch_rewards_sum/P_GAMES
        if true_max_reward_so_far < max_reward_so_far:
            true_max_reward_so_far = max_reward_so_far
        print("Make it train... after batch : ", batch)
        print("Reward mean = ", reward_mean)
        print("Max Batch reward so far: ", true_max_reward_so_far)
        print("batch: ", batch, "\nbatch_rewards: ", batch_rewards)
        discounted_batch_rewards_norm = PG.learn()
            
    PG.plot_cost()
