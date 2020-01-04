# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2)) 
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]         
        #print('self.state_space',self.state_space)
        self.state_init = random.choice(self.state_space)   
        # Start the first round
        self.reset()
        
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = [0 for x in range(m+t+d)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        
        return state_encod


#     # Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
         
#     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
 
#         state_encod = [0 for x in range(m+t+d+m+m)]
#         state_encod[state[0]] = 1
#         state_encod[m+state[1]] = 1
#         state_encod[m+t+state[2]] = 1
 
#         if (action[0] != 0):
#             state_encod[m+t+d+action[0]] = 1
#         if (action[1] != 0):
#             state_encod[m+t+d+m+action[1]] = 1
 
#         return state_encod       
 
    ## Getting number of requests

    def requests(self, state):
        
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        location = state[0]
        requests =0
        
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15            
#         print('requests',requests)        
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0] 
        actions = [self.action_space[i] for i in possible_actions_index]
    
        return actions,possible_actions_index

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        penalty = 0
        
        
        
        passenger_time = 0
        transit_time = 0
        
        curr_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        curr_time = state[1]
        curr_day = state[2]
        
        if ((pickup_loc== 0) and (drop_loc == 0)):  
            passenger_time=0
            transit_time=0
            penalty = 1
        elif (curr_loc == pickup_loc): 
            passenger_time = int(Time_matrix[curr_loc][drop_loc][curr_time][curr_day])
            transit_time=0
        else:
            transit_time = int(Time_matrix[curr_loc][pickup_loc][curr_time][curr_day])
            new_time = transit_time + curr_time
            if new_time >= 24:
                new_time = (new_time - 24 )%24
                new_day =  (curr_day + 1) %7               
            else:
                new_day = curr_day            
            passenger_time = int(Time_matrix[pickup_loc][drop_loc][new_time][new_day])
        
        reward = (R * passenger_time) - (C * transit_time) - (penalty * C)
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
#         total_time   = 0
#         transit_time = 0    
#         wait_time    = 0    
#         ride_time    = 0 

        new_time = 0
        new_loc = 0
        new_day = 0
       
        curr_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        curr_time = state[1]
        curr_day = state[2]
        
        if ((pickup_loc== 0) and (drop_loc == 0)):  
            
            new_time = 1 + curr_time            
            if new_time >= 24:
                new_time = (new_time - 24 )%24
                new_day =  (curr_day + 1) %7
            else:
                new_day = curr_day
            new_loc = curr_loc  
            
        elif (curr_loc == pickup_loc): 
            
            ride_time = int(Time_matrix[curr_loc][drop_loc][curr_time][curr_day])
            new_time = ride_time + curr_time
            if new_time >= 24:
                new_time = (new_time - 24 )%24
                new_day =  (curr_day + 1) %7
            else:
                new_day = curr_day 
            new_loc = drop_loc
                    
        else:
            #time taken to reach pickup
            transit_time = int(Time_matrix[curr_loc][pickup_loc][curr_time][curr_day])            
            new_time = transit_time + curr_time
            
            if new_time >= 24:
                new_time = (new_time - 24 )%24
                new_day =  (curr_day + 1) %7          
            else:
                new_day = curr_day
            
            #time taken to reach destination
            ride_time = int(Time_matrix[pickup_loc][drop_loc][new_time][new_day])
            
            new_time = new_time + ride_time
            
            if new_time >= 24:
                new_time = (new_time - 24 )%24
                new_day =  (curr_day + 1) %7
                
            new_loc = drop_loc
            
        return [new_loc,new_time,new_day] 


    def step(self,state, action, Time_matrix):
        
        next_state = self.next_state_func(state, action, Time_matrix)  
        rewards = self.reward_func(state, action, Time_matrix)
        
        return next_state,rewards

    def reset(self):
        return self.action_space, self.state_space, self.state_init
