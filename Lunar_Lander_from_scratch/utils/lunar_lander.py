#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:53:12 2023

@author: michiundslavki
"""

import numpy as np
import math
import gym
from gym.spaces import Box, Discrete
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import gym
from gym.spaces import Box, Discrete
import random
import copy
from torch import nn as nn
from torch.optim import AdamW
#from tqdm import tqdm
from scipy.integrate import odeint
from numpy import random

class Lunar_Lander_2(gym.Env):
    
       # metadata = {'render.modes': ['human']}
        def __init__(self, render_mode='human'):
            super( Lunar_Lander_2, self).__init__()
            self.render_mode = render_mode
            self.action_space = Box(low=0., high=10., shape= (2,)) #pass, make a cross at the first or second round
            self.observation_space = Box(low=0., high=1000., shape= (9,))
            self.state = np.array([3, 0, 5, 0, np.pi/6, 0, 5, 0, 0], dtype=float) # x, x_dot, y, y_dot, alpha, alpha_dot, x_land, y_land, foot_l, foot_r
            self.sim_state = np.array([2, 0, 20, 0, 0, 0])
            self.current_step = 1
            self.DT = np.array([0., 0.1])
            self.lander = lander(4, 3, 1.936, 1, 10000)
            self.center_mass_results = []
            info = {}
            self.rewards = 0
        def reset(self):
            self.state = np.array([4-2*np.random.rand(), 0, 10-2*np.random.rand(), 
                                  0, 0.1-0.2*np.random.rand(), 0, 5, 0, 0], dtype=float)
            #np.array([2, 0, 10, 0, 0, 0, 5, 0, 0], dtype=float)
            #np.array([5-10*np.random.rand(), 1-2*np.random.rand(), 20-2*np.random.rand(), 
                                 # 0, 0.2-0.4*np.random.rand(), 0, 5, 0, 0], dtype=float)#
            # # x, x_dot, y, y_dot, alpha, alpha_dot, x_land, y_land, foot_l, foot_r
            self.sim_state = self.sim_state[0:6]
            self.feet = self.lander.get_feet()
            self.current_step = 1
            self.DT = np.array([0., 0.1])
            self.center_mass_results = []
            self.rewards = 0
            return self.state
        def step(self, action):
            done = False
            max_times_steps = 50
            max_xvel = 0.5 # m/s
            max_yvel = 0.5 # m/s
            dt = 0.1 # sec 
            self.sim_state = self.state[0:6] # could do a function for that
            feet = self.lander.get_feet()
            
                #here comes a function that based on the sim state and the action 
            #print(action) 
            self.simulate_state(action*10000) #action*50000
            self.state[0:6] = self.sim_state
            feet = self.rot_trans(feet[0,:], feet[1,:])
            self.touch_down(np.array(feet))
            
            reward, done = self.feedback(max_times_steps, max_xvel, max_yvel)
   
    
            self.center_mass_results.append([self.state[0], self.state[2]])
    
            if self.current_step == max_times_steps: #after 1000 steps we stop this must be in accordance with the time in sim
                done = True
                reward = -5 # need to punish him 
            self.current_step += 1
            self.DT += dt
            self.rewards += reward
            #print(self.rewards)
            return self.state, reward, done, {}
        
        # the function that rotates and translate according to the center of mass motion 
        def rot_trans(self, x, y):
            x_prime = []
            y_prime = []
            A = np.array([[np.cos(self.state[4]), -np.sin(self.state[4])],
                  [np.sin(self.state[4]), np.cos(self.state[4])]])
            for i in range(0,len(x)):
                X = np.array([x[i], y[i]])
                x_prime.append(np.matmul(A[0], X) + self.state[0]) 
                y_prime.append(np.matmul(A[1],X) + self.state[2])
            x_prime = np.array(x_prime) 
            y_prime = np.array(y_prime)
            return x_prime, y_prime
        
        def simulate_state(self, force):
            state = odeint(lunar_lander, self.sim_state, self.DT, args=(force,), full_output=True)
            if (state[-1]['message'] == "Excess work done on this call (perhaps wrong Dfun type)."):
                 self.sim_state = self.sim_state
                 print(state[-1]['message'])
            else:
                self.sim_state = state[0][1,:]
        
        def touch_down(self, feet):
            eps = 0.1 # toleranz for the touch down in m
            if 0. < feet[1,0] < eps:
                self.state[7] = 1
            if 0. < feet[1,1] < eps:
                self.state[8] = 1
        def dist_target(self):
            dist = np.sqrt((self.state[6]-self.state[0])**2 + (0-self.state[2])**2)
            return dist
        
        def feedback(self, max_times_steps, max_xvel, max_yvel):
            fuel = 1-self.current_step/(max_times_steps+1) #-(1 - (1 - self.current_step/(max_times_steps+1)))
            reward = 0
            dist = self.dist_target()
            #reward = 0.1*((1-(dist/36))**2) #first version
            reward = -0.5*(dist/36)**2 #-0.5 # secon version
            done = False
            #if (self.state[6]-3 < self.state[0] < self.state[6]-3) and (bool(self.state[7]) and bool(self.state[8])):
            #      reward = 1
            #      done =True
            #if (self.state[5]<0.3 or  self.state[5]>5.9): #+/- 20°
            #    reward = 0.2*(dist/36)**2
            if self.state[2] < 0:
                reward = -2 #-5
                done = True
            if self.state[2] > 30:
                reward = -2
                done = True

            #if (bool(self.state[7]) or bool(self.state[8]) and abs(self.state[1])>max_xvel):
            #    reward = -20
            #    done = True
            #if (bool(self.state[7]) or bool(self.state[8]) and abs(self.state[3])>max_yvel):
            #    reward = -20
                

            #if (bool(self.state[7]) or bool(self.state[8]) and (abs(self.state[3])>max_yvel) and abs(self.state[1])>max_xvel):
            #    reward = 10
            #    done = True
            #if self.state[2] > 40:
            #    done = True
            if (self.state[0] < (self.state[6]-20)) or (self.state[0] > (self.state[6]+20)):
                done = True
                reward = -2 #last -10
                
            if bool(self.state[7]) and bool(self.state[8]):
                
                if (abs(self.state[1])<max_xvel and abs(self.state[3])<max_yvel):
                    reward = 100*fuel
                    done = True
                else:
                    reward = 5#last -5, -50 10 working
                    done=True
            return reward, done
                
        def render(self):
            feet = self.lander.get_feet()
            feet = self.rot_trans(feet[0,:], feet[1,:])
            print(feet)
            fig, ax = plt.subplots()
            center_mass = np.array(self.center_mass_results)
            ax.scatter(center_mass[:,0], center_mass[:,1] )
               # the landing plaform and the lander body
            lander_body_x, lander_body_y = self.rot_trans(self.lander.body[0, :], self.lander.body[1, :])
            
            landing_platform = ax.scatter([self.state[6]+5, self.state[6]-5], [0,0])
            ax.scatter(lander_body_x, lander_body_y)
            ax.grid()
            ax.axis('equal')
            return plt.show()

def lunar_lander(state, t, force):
    '''
    formulation with additional forces of the thrusters, the g Force, Newton friction and some trublente air
        only the friction of the center of mass motion is respected, as not high angular velocites and acceleration 
        are expected 
    '''
  
    g = -9.81 # m/s2
    a = 4 #m
    c = 3 #m
    h = 1.936 #m
    d = 2 # m
    M = 10000 #kg
    I = (a**2+h**2)*M/12 # sqaure
    # I = 240000  #trapez 
    cw = 1
    rho_air = 1.2 # kg/m^3
    A = a*d # m^2
    #turbulence = 0.2-0.4*np.random.rand()
    
    x = state[0] # is the angle
    x_dot = state[1] # is the angular veloicity
    dtx = x_dot
    dtx_dot = (1/M)*(force[0]+force[1])*math.sin(state[4]) - (1/M)*(cw*rho_air*A/2)*(state[1])**2#-turbulence)**2
    y = state[2] # is the angle
    y_dot = state[3] # is the angular veloicity
    dty = y_dot
    dty_dot = (1/M)*(force[0]+force[1])*math.cos(state[4]) + g - (1/M)*(cw*rho_air*A/2)*(state[3])**2#+turbulence)**2
    alpha = state[4] # is the angle
    alpha_dot = state[5] # is the angular veloicity
    dtalpha = alpha_dot
    dtalpha_dot = (1/I)*(-(a/2)*force[0]+(a/2)*force[1])
    
    dtstate_dt = [dtx, dtx_dot, dty, dty_dot, dtalpha, dtalpha_dot]
    # put to lander class 
    return dtstate_dt

class lander():
    
    def __init__(self, a, c, h, f, M):
        self.a = a
        self.c = c
        self.h = h
        self.foot = f
        self.M = M
        self.ys = h/3 * (a+2*c)/(a+c)
        self.body = np.array([[-self.a/2, -self.a/2, -self.a/4, self.a/4, self.a/2, 
                               self.a/2, self.a/2 - (self.a/2-self.c/2)/2, self.c/2,
                              0., -self.c/2, -(self.a/2 - (self.a/2-self.c/2)/2)],
                              [-self.ys-self.foot, -self.ys, -self.ys, -self.ys, 
                               -self.ys,-self.ys-self.foot, self.h/2-self.ys, self.h-self.ys,
                              self.h-self.ys, self.h-self.ys, self.h/2-self.ys]], 
                             dtype=object)
        #self.I

        # get the foot and body coordinate in the center of Mass system
        #def body(self):
        
        
    def render_it(self):
         plt.scatter(self.body[0,:], self.body[1,:])
    
    def get_feet(self):
        feet = np.array([[-self.a/2, self.a/2],[-self.ys-self.foot, -self.ys-self.foot]])
        
        return feet
    
    def get_Inertia(self):
        Ms = (self.M/len(self.body)) * np.ones(len(self.body))
        Inertia = 0 
        for i in range(0,len(self.body)):
            Inertia += Ms[i] * np.sqrt(self.body[0,i]**2 + self.body[1,i]**2)
         
        return Inertia               
                                
# possibly test the trained agent
    
if __name__ == '__main__' :
     
    test_lunar2 = Lunar_Lander_2()
    test_lunar2.reset()
    done = False
    while not done:
        action = np.array([8.5,8.5])#test_lunar2.action_space.sample()
        state, reward, done, _ = test_lunar2.step(action)
        print(action, state[2], state[3], reward)
    print(test_lunar2.rewards)
    test_lunar2.render()

    
    
    
    
    
    
    
    
    