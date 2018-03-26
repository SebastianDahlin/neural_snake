import math
import numpy as np
import random
import string
from operator import add


mutation_rate = 25

def nonlin(x):
    return 1/(1+np.exp(-x))

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
    
    # New way of doing it
    # check for walls
    # if snake_head[1]-1 == -1 and snake.move != [0, 1]: #Up
    #     move_list.append(1)
    # elif snake_head[1]+1 == snake.Y and snake.move != [0, -1]: #Down
    #     move_list.append(-1)
    # else:
    #     move_list.append(0)
    # if snake_head[0]-1 == -1 and snake.move != [1, 0]: #Left
    #     move_list.append(1)
    # elif snake_head[0]+1 == snake.X and snake.move != [-1, 0]: #Right
    #     move_list.append(-1)
    # else:
    #     move_list.append(0)
    # # check for own snake body
    # if [snake_head[0], snake_head[1]-1] in snake.whole and snake.move != [0, 1]: #Up
    #     move_list.append(1)
    # else:
    #     move_list.append(0)
    # if [snake_head[0], snake_head[1]+1] in snake.whole and snake.move != [0, -1]: #Down
    #     move_list.append(1)
    # else:
    #     move_list.append(0)
    # if [snake_head[0]-1, snake_head[1]] in snake.whole and snake.move != [1, 0]: #Left
    #     move_list.append(1)
    # else:
    #     move_list.append(0)
    # if [snake_head[0]+1, snake_head[1]] in snake.whole and snake.move != [-1, 0]: #Right
    #     move_list.append(1)
    # else:
    #     move_list.append(0)

    # New new
    #Looking at the whole play grid
    # for i in range(0, snake.X):
    #     for j in range(0, snake.Y):
    #         if [i,j] in snake.whole:
    #             move_list.append(1)
    #         else:
    #             move_list.append(0)


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
    def __init__(self, layer,typology):
        self.dendrons = [] # List to hold all ingoing connections to the neuron
        self.output = 0.0 # Used to sum the input * the sigmoid
        self.bias = 0
        if layer is None: # If the neuron is in the first layer, it will have no connections to former layer
            pass
        else:
            self.bias = random.uniform(-2,2)
            for neuron in layer: # Only for neurons not in the first layer
                con = Connection(neuron) 
                self.dendrons.append(con)

class Network():
    def __init__(self, typology): # Initiate the network with chosen typology
        #random.seed()
        self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
        self.fitness = 0
        self.fitness_history = []
        self.layers = [] # Holds neuron objects for each layer
        for neurons in typology: # Neurons equals number of neurons in each layer
            layer = [] # A list to hold all generated neuron objects
            for i in range(neurons): # Loop for each neuron in neurons
                if (len(self.layers)==0): # Check if this is the first layer column in the network
                    layer.append(Neuron(None,typology)) # If it is the first, add a "None" neuron.
                else: # If neuron is not in the first layer
                    layer.append(Neuron(self.layers[-1],typology)) # Send the amount of neurons in the prior layer into the connection creation class
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
                neuron.output = neuron.bias
                inst = 0
                for dDrons in neuron.dendrons:
                    neuron.output += self.layers[lay_inst][inst].output * dDrons.weight
                    inst += 1
                    # print("["+str(lay_inst)+","+str(inst)+" "+str(neuron.output)+"]")
                ##Sigmoid
                #print(neuron.output)
                neuron.output = nonlin(neuron.output)
                #print(neuron.output)
                ##Tanh
                #neuron.output = np.tanh(neuron.output)
            lay_inst += 1

    def calculate_output(self,snake):
        # Calculating X and Y (old way)
        # x_sum = self.layers[-1][0].output
        # y_sum = self.layers[-1][1].output
        # if abs(x_sum) > abs(y_sum):
        #     if abs(x_sum) > 0.3:
        #         #print("moving x: %s" %(x_sum))
        #         if x_sum > 0 and snake.move != [-1, 0]:
        #             snake.move = [1,0]
        #         if x_sum < 0 and snake.move != [1, 0]:
        #             snake.move = [-1,0]
        # if abs(x_sum) < abs(y_sum):
        #     if abs(y_sum) > 0.3:
        #         #print("moving y: %s" %(y_sum))
        #         if y_sum > 0 and snake.move != [0, -1]:
        #             snake.move = [0,1]
        #         if y_sum < 0 and snake.move != [0, 1]:
        #             snake.move = [0,-1]

        #Calculating up, down, left, right output new way
        directions = [[0,-1], [0,1], [-1,0], [1,0]]
        decision_list = []
        i = 0
        for y in directions:
            decision_list.append([y,self.layers[-1][i].output])
            i += 1
        decision_list = sorted(decision_list,key=lambda x: x[1], reverse=True)
        #print(decision_list)
        decision = decision_list[0] #Creates for example [[0,1], 0.6]
        reverse_decision = [i * -1 for i in decision[0]]
        if decision[1] > 0.5 and snake.move != reverse_decision:
            snake.move = decision[0]



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

    def perform_back_propagation(self, input_list, output_list, real_output_list):
        y = np.array(output_list)
        last_layer = np.array(real_output_list)
        last_layer_error = y - last_layer
        #print(last_layer_error)
        #Perform back propagation
        last_layer_delta = last_layer_error*nonlin(last_layer)
        # Perform the multiplication
        inst = 0
        for neuron in self.layers[-1]:
            for dDrons in neuron.dendrons:
                dDrons.weight = dDrons.weight + last_layer_delta[inst] * input_list[inst]
            inst += 1
        #print("here is the last layer delta %a" % (last_layer_delta))