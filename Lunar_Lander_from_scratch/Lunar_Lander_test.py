#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:00:10 2023

@author: michiundslavki
"""
import numpy as np
import os
import torch.nn as nn
from utils.lunar_lander import Lunar_Lander_2
from stable_baselines3 import DDPG, TD3
#from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#creating the RL environment as an instance of the Lunar_Lander class
env = Lunar_Lander_2()
env.reset()
 #random agent
done = False
while not done:
      action = env.action_space.sample() #np.array([4,4]) 
      state, reward, done, _ = env.step(action)
#      print(state[2], state[3], reward)
#      print(test_lunar2.rewards)
env.render()



#trained agent
MODEL_PATH = "./models/DDPG/5000_reward_Model_4_test.zip"
model = DDPG.load(MODEL_PATH)
#results_plotter.plot_results(["./log"], 10e5, results_plotter.X_TIMESTEPS, "Breakout")



obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(action, obs, rewards)
env.render()