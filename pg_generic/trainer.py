# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    trainer.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:55:13 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/23 11:08:47 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src import game
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

# Policy gradient has high variance, seed for reproducability
#env.seed(1)

RENDER_ENV = True
BATCHES = 30
P_GAMES = 5
STEPS = 100
rewards = []
RENDER_REWARD_MIN = 100

R = 6
C = 7
X_DIM = R * C * 2 + 5
ACTIONS = ["right", "down", "left", "up", "toggle"]

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
    load_path = "./output/weights/pizza-temp.ckpt"
    save_path = "output/weights/pizza-temp.ckpt"

    PG = PolicyGradient(
            n_x = X_DIM,
            n_y = 5,
            learning_rate=0.01,
            reward_decay=0.95,
            load_path=load_path,
            save_path=save_path
            )

    for batch in range(BATCHES):
        for p_game in range(P_GAMES):
            env = game.Game({'max_steps': 100})
            episode_reward = 0
            h = 5			
            l = 1
            pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
            pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            state = env.init(pizza_config)[0]
	    #np.zeros(OBSERVATION_DIM) #get only first value of tuple
            for step in range(STEPS):
                if RENDER_ENV: 
                    env.render()
                # sample one action with the given probability distribution
                # 1. Choose an action based on observation
                action = PG.choose_action(state)

                # 2. Take action in the environment
                state_, reward, done, info = env.step(ACTIONS[action])

                # 3. Store transition for training
                PG.store_transition(preprocess(state), action, reward)
            
                # Save new state
                #state = state_
                if done:
                    episode_rewards_sum = sum(PG.episode_rewards)
                    rewards.append(episode_rewards_sum)
                    max_reward_so_far = np.amax(rewards)

                    print("==========================================")
                    print("p_game: ", p_game)
                    print("batch: ", batch)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)

            # 4. Train neural network
        print("Make it train... after batch :") #p_game)
        discounted_episode_rewards_norm = PG.learn()
           
    PG.plot_cost()
