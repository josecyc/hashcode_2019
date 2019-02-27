# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    trainer.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/22 21:55:13 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/27 10:40:09 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src import game_b
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import boards as bd
from time import gmtime, strftime

# Policy gradient has high variance, seed for reproducability
#env.seed(1)

RENDER_ENV = True
BATCHES = 2000
P_GAMES = 750
STEPS = 150
rewards = []
batch_rewards = []
game_scores = []
true_max_reward_so_far = 0
Learning_rate = 0.002
GAMMA = 0.95

R = 6
C = 7
X_DIM = R * C * 3 + 3
ACTIONS = ["right", "down", "left", "up", "cut_right", "cut_down", "cut_up", "cut_left"]

if __name__ == "__main__":

    # Load checkpoint
    s = strftime("./output/weights/%a_%d_%b_%Y_%H:%M/pizza.ckpt", gmtime())
    load_path = None #"./output/weights/pizza-temp.ckpt"
    save_path = s

    PG = PolicyGradient(
            n_x = X_DIM,
            n_y = 8,
            learning_rate=Learning_rate,
            reward_decay=GAMMA,
            steps=STEPS,
            load_path=load_path,
            save_path=save_path
            )

    for batch in range(BATCHES):
        for p_game in range(P_GAMES):
            env = game_b.Game({'max_steps': 100})
            batch_reward = 0
            #h = 6			
            #l = 2
            #pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
            #pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            pizza_config = bd.rand_pizza(6, 7)
            state = env.init(pizza_config)[0]
            for step in range(STEPS):
                # 1. Choose an action based on observation
                state = bd.preprocess(state)
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
            max_reward_so_far = np.amax(game_scores)
            PG.game_rewards = []
            if p_game == (P_GAMES//2):
                print("==========================================")
                print("\nMIDDLE OF THE BATCH game: ", p_game)
                env.render()
                print("==========================================")
        if true_max_reward_so_far < max_reward_so_far:
            print("\nNEW MAX REWARD:", max_reward_so_far, "\n")
            print("==========================================")
            env.render()
            true_max_reward_so_far = max_reward_so_far
            print("==========================================")
        print("==========================================")
        print("len game_scores", len(game_scores))
        print("game_scores\n", game_scores)
        reward_mean = sum(game_scores)/P_GAMES
        game_scores = []
        print("==========================================")
        print("FINAL GAME OF BATCH")
        print("Training...")
        print("Game reward mean = ", reward_mean)
        print("Max Batch reward so far: ", true_max_reward_so_far)
        print("L = ", bd.L)
        print("H = ", bd.H)
        print("BATCH: ", batch, " out of ", BATCHES) 
        print("GAMES per BATCH:", P_GAMES)
        print("Learning rate: ", Learning_rate)
        print("Gamma: ", GAMMA)
        print("Neurons: ", PG.neurons_layer_1)

        env.render()
        # 4. Train neural network
        discounted_batch_rewards_norm = PG.learn()
        print("==========================================")
        print("VALIDATION\n")
        bd.run_validation(PG, STEPS)
        print("==========================================")
