# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    trainer.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:55:13 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/26 13:59:26 by jcruz-y-         ###   ########.fr        #
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
P_GAMES = 250
STEPS = 100
rewards = []
batch_rewards = []
game_scores = []
RENDER_REWARD_MIN = 100
true_max_reward_so_far = 0

R = 6
C = 7
X_DIM = R * C * 3 + 3
ACTIONS = ["right", "down", "left", "up", "cut_right", "cut_down", "cut_up", "cut_left"]

def preprocess(state_dict):
    cursor_map = np.zeros(np.array(state_dict['ingredients_map']).shape)
    #print(state_dict['cursor_position'])
    cursor_map[state_dict['cursor_position']] = 1
    #print(cursor_map)
    #print("INPUT CURSOR MAP:\n", cursor_map.ravel())
    state = np.concatenate((
            np.array(state_dict['ingredients_map']).ravel(),
            np.array(state_dict['slices_map']).ravel(),
            cursor_map.ravel(),
      #      np.array(state_dict['cursor_position']).ravel(),
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
            h = 6			
            l = 2
            pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
            pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            state = env.init(pizza_config)[0]
	    #state[0] #get only first value of tuple
            for step in range(STEPS):
                #if RENDER_ENV: 
                #    env.render()
                # 1. Choose an action based on observation
                state = preprocess(state)
                #print("NEW STATE\n", state)
                action = PG.choose_action(state)

                # 2. Take action in the environment
                state_, reward, done, info = env.step(ACTIONS[action])

                # 3. Store transition for training
                PG.store_transition(state, action, reward)
            
                # Save new state
                state = state_
                if done:
                    break
            game_score = sum(PG.game_rewards)
            #print("game_score", game_score)
            game_scores.append(game_score)
            #print("game_scores", game_scores)
            #batch_rewards_sum = sum(PG.batch_rewards)
            #print("batch_rewards_sum", batch_rewards_sum)
            #rewards.append(batch_rewards_sum)
            #print("rewards", rewards)
            #print("partial reward mean", batch_rewards_sum/(p_game + 1))
            max_reward_so_far = np.amax(game_scores)
            #print("max_reward_so_far", max_reward_so_far)

            #print("==========================================")
            #print("p_game: ", p_game)
            #print("batch: ", batch)
            #print("Reward: ", episode_rewards_sum)
            #print("Max Batch reward so far: ", max_reward_so_far)
            PG.game_rewards = []
            #print("game: ", p_game, )
            # 4. Train neural network
        #reward_mean = batch_rewards_sum/P_GAMES
        if true_max_reward_so_far < max_reward_so_far:
            print("\nNEW MAX REWARD:", max_reward_so_far, "\n")
            env.render()
            true_max_reward_so_far = max_reward_so_far
        print("\n\nlen game_scores", len(game_scores))
        print("game_scores", game_scores)
        reward_mean = sum(game_scores)/P_GAMES
        game_scores = []
        print("Make it train... after batch : ", batch)
        print("Game reward mean = ", reward_mean)
        print("Max Batch reward so far: ", true_max_reward_so_far)
        print("batch: ", batch)
        env.render()
        discounted_batch_rewards_norm = PG.learn()
            
    #PG.plot_cost()
