import math
import numpy as np
import random
import string
from operator import add
import numpy as np
import time

mutation_rate = 25

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def set_move(snake):
    snake_head = snake.whole[0]
    move_list = []
    # Old way of doing it:
    # #check for obstacles
    if snake_head[1]-1 == -1 or [snake_head[0], snake_head[1]-1] in snake.whole and snake.move != [0, 1]: #Up
        move_list.append(1)
    else:
        move_list.append(0)
    if snake_head[1]+1 == snake.Y or [snake_head[0], snake_head[1]+1] in snake.whole and snake.move != [0, -1]: #Down
        move_list.append(1)
    else:
        move_list.append(0)
    if snake_head[0]-1 == -1 or [snake_head[0]-1, snake_head[1]] in snake.whole and snake.move != [1, 0]: #Left
        move_list.append(1)
    else:
        move_list.append(0)
    if snake_head[0]+1 == snake.X or [snake_head[0]+1, snake_head[1]] in snake.whole and snake.move != [-1, 0]: #Right
        move_list.append(1)
    else:
        move_list.append(0)

    #Also add suggestion moves to get the apple
    # New way
    sug_ver, sug_hor = 0, 0
    sug_right, sug_left, sug_down, sug_up = 0,0,0,0
    sug_x = snake.apple[0] - snake_head[0]
    if sug_x > 0:
        sug_right = 1
        sug_hor = 1
    elif sug_x < 0:
        sug_left = 1 
        sug_hor = -1
    sug_y = snake.apple[1] - snake_head[1]
    if sug_y > 0:
        sug_down = 1
        sug_ver = 1
    elif sug_y < 0:
        sug_up = 1
        sug_ver = -1
    move_list.extend((sug_up,sug_down,sug_left,sug_right))
    #move_list.extend((sug_hor,sug_ver))
    x = np.array(move_list)
    return x.reshape(x.shape[0], 1)

class Network():
    '''This class hold the entire network'''
    def __init__(self, topology):
        self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
        self.topology = topology
        self.fitness = 0
        self.fitness_history = []
        self.layers = []
        self.weights = []
        self.biases = []
        for layer_count in topology:
            layer = np.zeros(layer_count)
            self.layers.append(layer)
        
        for i in range(0,len(topology)-1):
            # Create the weights
            weight = np.random.normal(0, 1, (topology[i+1],topology[i]))
            self.weights.append(weight)
            # Create the biases
            bias = np.random.random((topology[i+1], 1))
            self.biases.append(bias)

    def forward_propagate(self, obs_list):
        self.layers[0] = obs_list
        for i in range(0,len(self.topology)-1): # Change this index
            self.layers[i+1] = sigmoid(self.weights[i].dot(self.layers[i]) + self.biases[i])
        
    def new_name(self):
        self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

    def calculate_output(self,snake):
        #Calculating up, down, left, right output new way
        directions = [[0,-1], [0,1], [-1,0], [1,0]]
        decision_list = []
        i = 0
        for y in directions:
            decision_list.append([y,self.layers[-1][i]])
            i += 1
        decision_list = sorted(decision_list,key=lambda x: x[1], reverse=True)
        #print(decision_list)
        decision = decision_list[0] #Creates for example [[0,1], 0.6]
        reverse_decision = [i * -1 for i in decision[0]]
        if decision[1] > 0.5 and snake.move != reverse_decision:
            snake.move = decision[0]

    def cross_over(self, net1, net2, round_nr):
        # Weights
        for weights1, weights2 in zip(net1.weights, net2.weights):
            for weight1, weight2 in zip(weights1, weights2):
                for one_weight, two_weight in zip(weight1, weight2):
                    one_weight = random.choice([one_weight,two_weight])

        # Biases
        for biases1, biases2 in zip(net1.biases, net2.biases):
            for bias1, bias2 in zip(biases1, biases2):
                bias1 = random.choice([bias1, bias2])
                    

        