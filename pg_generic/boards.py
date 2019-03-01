# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    boards.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/26 19:19:52 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/27 18:56:14 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import random
import numpy as np
import src.game_b as game

#r = random.randint(3, 1000)
#c = random.randint(3, 1000)
#l = random.randint(1, r/2) 
#minim_max = l * 2 
#h = random.randint(minim_max, r)
#R = 20
#C = 20
H = 6
L = 2

ACTIONS = ["right", "down", "left", "up", "cut_right", "cut_down", "cut_up", "cut_left"]

def preprocess(state_dict):
    cursor_map = np.zeros(np.array(state_dict['ingredients_map']).shape)
    #flat_ing_map = np.array(state_dict['ingredients_map']).ravel()
    #ing_map_len = len(flat_ing_map)
    #ing_map_flat_extended = flat_ing_map + np.zeros(1000000 - ing_map_len)
    cursor_map[state_dict['cursor_position']] = 1
    state = np.concatenate((
            np.array(state_dict['ingredients_map']).ravel(),
            #ing_map_flat_extended,
            np.array(state_dict['slices_map']).ravel(),
            cursor_map.ravel(),
            [state_dict['slice_mode'],
            state_dict['min_each_ingredient_per_slice'],
            state_dict['max_ingredients_per_slice']],
	))
    return state.astype(np.float).ravel()

def rand_pizza(r, c):
   #h = random.randint(2, (r * c) // 5)
   #l = random.randint(1, 3)
    #r = random.randint(2, 1000)
    #c = random.randint(2, 1000)
    l = random.randint(1, r/2) 
    minim_max = l * 2
    h = random.randint(minim_max, r)
    ing = ['T', 'M']
    pizza = []
    for _ in range(r):
        ls = []
        for _ in range(c):
            ls.append(ing[random.randint(0,1)])
        pizza.append(''.join(ls))
    return {'pizza_lines' : pizza, 'r' : r, 'c' : c, 'l' : l, 'h' : h}


def run_validation(PG, steps):
    env = game.Game({'max_steps': steps})
    R = 6
    C = 7
    h = 6
    l = 2
    pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
    #pizza_lines = ["TMTMTMT","MTMTMTM", "TMTMTMT", "MTMTMTM", "TMTMTMT", "MTMTMTM","TMTMTMT"]
    pizza_config = {'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h}
    state = env.init(pizza_config)[0]  # np.zeros(OBSERVATION_DIM) #get only first value of tuple
    done = False
    acts = []
    for step in range(steps):
        state = preprocess(state)
        # 1. Choose an action based on observation
        action = PG.choose_action(state)

        # 2. Take action in the environment
        state_, reward, done, info = env.step(ACTIONS[action])
        acts.append(ACTIONS[action])
        state = state_
    print(acts)
    env.render()
    #game ends


