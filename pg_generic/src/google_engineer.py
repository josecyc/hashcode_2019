# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    google_engineer.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jcruz-y- <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/02/24 19:56:10 by jcruz-y-          #+#    #+#              #
#    Updated: 2019/02/27 21:10:55 by jcruz-y-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.pizza import Pizza, Direction
from src.ingredients import Ingredients

import numpy as np
import json

POSITIVE_REWARD = 1.0
NEUTRAL_REWARD  = 0
NEGATIVE_REWARD = -0

class ActionNotFoundException(Exception):
    pass

class GoogleEngineer:
    delta_position = {
        Direction.right: (0,1),
        Direction.down:  (1,0),
        Direction.left:  (0,-1),
        Direction.up:    (-1,0),
    }

    def __init__(self, pizza_config):
        self.pizza = Pizza(pizza_config['pizza_lines'])
        self.min_each_ingredient_per_slice = pizza_config['l']
        self.max_ingredients_per_slice = pizza_config['h']
        self.cursor_position = (0,0)
        self.slice_mode = False
        self.valid_slices = []
        self.score = 0

    def score_of(self, slice):
        if min(self.pizza.ingredients.of(slice)) >= self.min_each_ingredient_per_slice:
            return slice.ingredients
        return 0


    def move(self, direction):
        next_cursor_position = tuple(x0+x1 for x0,x1 in zip(self.cursor_position,self.delta_position[direction]))
        if (next_cursor_position[0] >= 0 and next_cursor_position[0] < self.pizza.r and
            next_cursor_position[1] >= 0 and next_cursor_position[1] < self.pizza.c):

            self.cursor_position = next_cursor_position
            return NEUTRAL_REWARD
        return NEGATIVE_REWARD

    def increase(self, direction):
        slice = self.pizza.slice_at(self.cursor_position)
        if slice is None:
            return -0.1
        new_slice = self.pizza.increase(slice, direction, self.max_ingredients_per_slice)
        if (new_slice is not None and min(self.pizza.ingredients.of(new_slice)) >=
            self.min_each_ingredient_per_slice):

            if slice in self.valid_slices:
                self.valid_slices.remove(slice)
            self.valid_slices.append(new_slice)
            score = self.score_of(new_slice) - self.score_of(slice)
            self.score += score
            return score * POSITIVE_REWARD
        return NEUTRAL_REWARD if new_slice is not None else NEGATIVE_REWARD

    def increase_neg(self, direction):
        slice = self.pizza.slice_at(self.cursor_position)
        #new_slice = self.pizza.increase(slice, direction, self.max_ingredients_per_slice)
        #if (new_slice is not None and min(self.pizza.ingredients.of(new_slice)) <
         #   self.min_each_ingredient_per_slice):
        if (slice is not None and min(self.pizza.ingredients.of(slice)) <
            self.min_each_ingredient_per_slice):
            #if slice in self.valid_slices:
             #   self.valid_slices.remove(slice)
            #self.valid_slices.append(slice)
            score = slice.ingredients #- self.score_of(slice)
            #self.score += score
            return POSITIVE_REWARD * 0.5 * score
        return NEUTRAL_REWARD #NEUTRAL_REWARD if new_slice is not None else NEGATIVE_REWARD

    def do(self, action):
        #cut = 0
        if action in ['cut_up', 'cut_left', 'cut_down', 'cut_right']:
            self.slice_mode = True
            action = 'up' if action == 'cut_up' else action
            action = 'down' if action == 'cut_down' else action
            action = 'left' if action == 'cut_left' else action
            action = 'right' if action == 'cut_right' else action
        elif self.slice_mode == True:
            self.slice_mode = False
            reward = -self.increase_neg(Direction[action])
            slice = self.pizza.slice_at(self.cursor_position)
            for direction in Direction:
                self.pizza.disable_increase_around(slice, direction, 1)
            self.move(Direction[action])
            return reward
        else:
            self.slice_mode = False
            #reward = -self.increase_neg(Direction[action])
            #return reward

        if action == 'toggle':
            self.slice_mode = not self.slice_mode
            return 0
        if action not in Direction.__members__:
            raise ActionNotFoundException('Action \'{}\' is not recognised.'.format(action))

        if self.slice_mode:
            reward = self.increase(Direction[action])
            self.cut = 1
            return reward
        reward = self.move(Direction[action])
        return reward

    def state(self):
        return {
            'ingredients_map': self.pizza.ingredients._map.tolist(),
            'slices_map': self.pizza._map.tolist(),
            'cursor_position': self.cursor_position,
            'slice_mode': self.slice_mode,
            'min_each_ingredient_per_slice': self.min_each_ingredient_per_slice,
            'max_ingredients_per_slice': self.max_ingredients_per_slice,
        }
