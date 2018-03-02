import math
import numpy as np
import random
import string
from operator import add


mutation_rate = 25
extended = False

def set_move(snake):
    snake_head = snake.whole[0]
    move_list = []
    if extended is False:
    #check for obstacles
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
    else:
        # check for walls
        if snake_head[1]-1 == -1 and snake.move != [0, 1]: #Up
            move_list.append(1)
        elif snake_head[1]+1 == snake.Y and snake.move != [0, -1]: #Down
            move_list.append(-1)
        else:
            move_list.append(0)
        if snake_head[0]-1 == -1 and snake.move != [1, 0]: #Left
            move_list.append(1)
        elif snake_head[0]+1 == snake.X and snake.move != [-1, 0]: #Right
            move_list.append(-1)
        else:
            move_list.append(0)
        # check for own snake body
        if [snake_head[0], snake_head[1]-1] in snake.whole and snake.move != [0, 1]: #Up
            move_list.append(1)
        else:
            move_list.append(0)
        if [snake_head[0], snake_head[1]+1] in snake.whole and snake.move != [0, -1]: #Down
            move_list.append(1)
        else:
            move_list.append(0)
        if [snake_head[0]-1, snake_head[1]] in snake.whole and snake.move != [1, 0]: #Left
            move_list.append(1)
        else:
            move_list.append(0)
        if [snake_head[0]+1, snake_head[1]] in snake.whole and snake.move != [-1, 0]: #Right
            move_list.append(1)
        else:
            move_list.append(0)

    #Also add suggestion moves to get the apple
    sug_ver = 0
    sug_hor = 0
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
    #move_list.extend((sug_up,sug_down,sug_left,sug_right))
    move_list.extend((sug_hor,sug_ver))
    #print(move_list)
    return(move_list)


class Connection:
    innovation = 0
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = random.uniform(-1,1)
        self.enabled = 1
        Connection.innovation += 1
        self.innov = Connection.innovation


class Neuron():
    def __init__(self, layer):
        self.dendrons = [] # List to hold all ingoing connections to the neuron
        self.output = 0.0 # Used to sum the input * the sigmoid
        if layer is None: # If the neuron is in the first layer, it will have no connections to former layer
            pass
        else:
            for neuron in layer: # Only for neurons not in the first layer
                con = Connection(neuron) 
                self.dendrons.append(con)

class Network():
    def __init__(self, typology): # Initiate the network with chosen typology
        #random.seed()
        self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
        self.fitness = 0
        self.layers = [] # Holds neuron objects for each layer
        for neurons in typology: # Neurons equals number of neurons in each layer
            layer = [] # A list to hold all generated neuron objects
            for i in range(neurons): # Loop for each neuron in neurons
                if (len(self.layers)==0): # Check if this is the first layer column in the network
                    layer.append(Neuron(None)) # If it is the first, add a "None" neuron.
                else: # If neuron is not in the first layer
                    layer.append(Neuron(self.layers[-1])) # Send the amount of neurons in the prior layer into the connection creation class
            self.layers.append(layer) # Append all new neurons, containing all connections to the netoworks layer in question
    
    def new_name(self):
        self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

    def update_output(self, obs_list):
        neurons = self.layers[0]
        for i in range(0,len(obs_list)):
            neurons[i].output =  obs_list[i]
        layers = iter(self.layers)
        next(layers)
        #inst = 0
        lay_inst = 0
        for layer in layers:
            for neuron in layer:
                neuron.output = 0.0
                inst = 0
                for dDrons in neuron.dendrons:
                    neuron.output += self.layers[lay_inst][inst].output * dDrons.weight
                    inst += 1
                    # print("["+str(lay_inst)+","+str(inst)+" "+str(neuron.output)+"]")
            lay_inst += 1

    def calculate_output(self,snake):
        # Calculating X and Y
        x_sum = self.layers[-1][0].output
        y_sum = self.layers[-1][1].output

        if abs(x_sum) > abs(y_sum):
            if abs(x_sum) > 0.5:
                #print("moving x: %s" %(x_sum))
                if x_sum > 0 and snake.move != [-1, 0]:
                    snake.move = [1,0]
                if x_sum < 0 and snake.move != [1, 0]:
                    snake.move = [-1,0]
        if abs(x_sum) < abs(y_sum):
            if abs(y_sum) > 0.5:
                #print("moving y: %s" %(y_sum))
                if y_sum > 0 and snake.move != [0, -1]:
                    snake.move = [0,1]
                if y_sum < 0 and snake.move != [0, 1]:
                    snake.move = [0,-1]

    def cross_over(self,net1, net2, round_nr):
        # Iterate layers of both nets
        if round_nr == 0:
            chance = 0
        else:
            chance = int(round_nr/mutation_rate)
        if chance == 0:
            chance = 2
        layers1 = iter(self.layers)
        lay_inst = 0
        for layer in layers1:
            #print(layer)
            inst = 0
            for neuron in layer:
                dInst = 0
                for dDrons in neuron.dendrons:
                    if chance < 2:
                        chance = random.randint(1,3)
                    else:
                        chance = random.randint(1,chance)
                    #print("Chance number: %s" %(chance))
                    if chance == 1:
                        pass
                    else:
                        chance2 = random.randint(1,2)
                        if chance2 == 1:
                            dDrons.weight = net1.layers[lay_inst][inst].dendrons[dInst].weight
                        if chance2 == 2:
                            dDrons.weight = net2.layers[lay_inst][inst].dendrons[dInst].weight
                        #print("Chanse weight %s" % (dDrons.weight))
                    dInst +=1
                    #print("Layer: " + str(lay_inst)+" neuron: " + str(inst)+ " dendron: " + str(dInst) + " weight: " +str(dDrons.weight) )
                inst += 1
            lay_inst += 1