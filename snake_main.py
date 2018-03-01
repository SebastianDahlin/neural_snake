'''
Self playing snake by evolving weights of decided typology neural net.
version: 0.1
author: Sebastian Dahlin
'''
import pygame
import random
import string
import time
import snake_neural
import copy
import sys

### Inputs for the run
# Set the tick number. Higher means faster snake.
set_tick = 50
# Screen size
X = 30
Y = 30
# Set the point goal. Until it is reached the GUI will not show the snake. 
# Set to 0 if you want to see it from start.
max_point_goal =45000
# Set the chance of mutation. Must be an integer bigger than 0. Bigger number means more mutation.
snake_neural.mutation_rate = 500
# Extended is for experimental purposes only
snake_neural.extended = True
## This is the geometry for the neural network: 
# If extended is false keep first to 6, last to 2.
# If extended is true keep first to 10, last to 2.
if snake_neural.extended is True:   
    topology = [8,16,4,2]
else:
    topology = [6,4,2]


class Snake():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.point = 0
        self.move = [1,0]
        self.last = [3,4]
        self.current = [4, 4]
        self.whole = [[4,4],[3,4],[2,4]]
        self.got_apple = False
        self.apple = self.get_apple_placement()
        self.fitness = 0
        self.fitness_since_last_apple = 0

    def iterate(self):
        self.last =  self.current
        self.current = [self.current[0]+self.move[0],self.current[1]+self.move[1]]
        if self.got_apple is False:
            self.whole = self.whole[:-1]
        else:
            self.got_apple = False
        self.whole.insert(0,self.current)
        self.fitness += 1
        self.fitness_since_last_apple +=1

    def get_apple_placement(self):
        apple = []
        for i in range(0, X):
            for j in range(0, Y):
                apple.append([i,j])
        for place in self.whole:
            apple.remove(place)
        return(random.choice(apple))

    def check_apple(self):
        if self.apple == self.current:
            self.apple = self.get_apple_placement()
            self.point += 1000
            self.got_apple = True
            self.fitness_since_last_apple = 0

    def check_game_over(self):
        if self.current[0] > X-1 or self.current[0] < 0:
            return(False)
        elif self.current[1] > Y-1 or self.current[1] < 0:
            return(False)
        elif self.current in self.whole[1:]:
            return(False)
        elif self.fitness_since_last_apple > 300:
            return(False)
        else:
            return(True)      

def render(snake, generation_no,run_no):
    if max_point > max_point_goal:
        SCREEN.fill((0, 0, 0))
        myfont = pygame.font.SysFont("monospace", 20)
        label = myfont.render("Generation: " + str(generation_no) + " Run no. " + str(run_no)+" Points: "+str(snake.point) +" Fitness: " +str(snake.fitness) +"  FSLA:  " + str(snake.fitness_since_last_apple), 1, (255,255,0))
        SCREEN.blit(label, (20, 20))
        for i in range(0, X):
            for j in range(0, Y):
                #Draw apple
                pygame.draw.rect(SCREEN, (255, 0, 0),(100+snake.apple[0]*30, 100+snake.apple[1]*20, 30, 20))
                #Draw grid
                pygame.draw.rect(SCREEN, (255, 255, 255),(100+i*30, 100+j*20, 30, 20),1)
                #Draw snake
                for item in snake.whole:
                    if item == [i,j]:
                        pygame.draw.rect(SCREEN, (255, 255, 255),(100+i*30, 100+j*20, 30, 20))
        pygame.display.flip()
    else:
        pass
    

#Set the screen size
screen_width = 1200
screen_height = 800

#Main program
pygame.init()
clock = pygame.time.Clock()
clock.tick(1)
SCREEN = pygame.display.set_mode((screen_width, screen_height))
#Set the title of the window
pygame.display.set_caption('Snake game')
#Below is the running game loop. When the snake dies, it should reset.
history = []
name_list = []
#First inistantiation
net_list = []
generation_no = 1
run_no = 1
local_max = 0
max_point = 0
for i in range(0,10):
    net_inst = snake_neural.Network(topology)
    net_list.append(net_inst)
while True:
    print("Generation %s with local max: %s" % (generation_no, local_max))
    for net in net_list:
        #print("Now running %s" % (net.name))
        do_again = True
        snake = Snake(X, Y)
        while do_again is True:
            render(snake, generation_no, run_no)
            net.update_output(snake_neural.set_move(snake))
            if max_point > max_point_goal:
                print(snake_neural.set_move(snake))
            net.calculate_output(snake)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    #Escape will quit the program
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            # #manual input
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         done = True
            # #pygame.event.pump()
            # for i in range(0, 30):
            #     keys = pygame.key.get_pressed()
            #     if keys[pygame.K_LEFT] and snake.move[0] == 0:
            #         snake.move = [-1,0]
            #     if keys[pygame.K_RIGHT] and snake.move[0] == 0:
            #         snake.move = [1,0]
            #     if keys[pygame.K_UP] and snake.move[1] == 0:
            #         snake.move = [0,-1]
            #     if keys[pygame.K_DOWN] and snake.move[1] == 0:
            #         snake.move = [0,1]
            snake.iterate()
            net.fitness = snake.fitness + snake.point
            snake.check_apple()
            do_again = snake.check_game_over()
            if max_point > max_point_goal:
                clock.tick(set_tick)
        history.append(net.fitness)
        name_list.append(net.name)
        run_no += 1
        if max_point > max_point_goal:
            print("died")

    # If a new max for points are reached, print it
    if max(history) > max_point:
        max_point = max(history)
        print(max_point)
    # Get the local max
    local_max = max(history)
    # Start printing more data if max_point_goal is reached
    if max_point > max_point_goal:
        for name, point in zip(name_list, history):
            print(str(name)+" got " + str(point)+" points.")

    # Save each neural net in a list with it's corresponding points
    record = [list(a) for a in zip(net_list, history)]

    # Sort the list from top scoring and ascending
    sorted_record = sorted(record,key=lambda x: x[1], reverse=True)

    # Remove the score column from the list
    sorted_list = [x[0] for x in sorted_record]

    # Create a new list with only the four best of the networks
    net_list = sorted_list[0:4]

    # Make copies of the four neural nets in net list
    c1 = copy.copy(net_list[0])
    c2 = copy.copy(net_list[1])
    c3 = copy.copy(net_list[2])
    c4 = copy.copy(net_list[3])

    # Make random copies of the winners
    c5 = copy.copy(net_list[random.randint(0,3)])
    c6 = copy.copy(net_list[random.randint(0,3)])
    c7 = copy.copy(net_list[random.randint(0,3)])
    c8 = copy.copy(net_list[random.randint(0,3)])
    c9 = copy.copy(net_list[random.randint(0,3)])
    c10 = copy.copy(net_list[random.randint(0,3)])
    c11 = copy.copy(net_list[random.randint(0,3)])
    c12 = copy.copy(net_list[random.randint(0,3)])

    # Make four more copies that can be directly transfered to the new list. Add new names
    c13 = copy.copy(net_list[0])
    c13.new_name()
    c14 = copy.copy(net_list[1])
    c14.new_name()
    c15 = copy.copy(net_list[2])
    c15.new_name()
    c16 = copy.copy(net_list[3])
    c16.new_name()
    top_cross = snake_neural.Network(topology)
    top_cross.cross_over(c1,c2,run_no)

    # Create three offsprings as crosssover between three random winners
    rand_cross1 = snake_neural.Network(topology)
    rand_cross1.cross_over(c5,c6,run_no)
    rand_cross2 = snake_neural.Network(topology)
    rand_cross2.cross_over(c7,c8,run_no)
    rand_cross3 = snake_neural.Network(topology)
    rand_cross3.cross_over(c9,c10,run_no)

    # Create two offsprings as a  copy of two random winners, just applying mutation
    rand_net1 = snake_neural.Network(topology)
    rand_net1.cross_over(c11, c11,run_no)
    rand_net2 = snake_neural.Network(topology)
    rand_net2.cross_over(c12, c12,run_no)

    # Also include a copy of the three winners once again
    # c17 = copy.copy(net_list[0])
    # c18 = copy.copy(net_list[1])
    # c19 = copy.copy(net_list[2])
    
    #append all new nets
    net_list.append(top_cross)
    net_list.append(rand_cross1)
    net_list.append(rand_cross2)
    net_list.append(rand_cross3)
    net_list.append(rand_net1)
    net_list.append(rand_net2)
    net_list.append(c13)
    net_list.append(c14)
    net_list.append(c15)
    net_list.append(c16)
    # net_list.append(c17)
    # net_list.append(c18)
    # net_list.append(c19)

    #Update the run number and clear history and name list
    generation_no += 1
    run_no = 0
    history = []
    name_list = []

