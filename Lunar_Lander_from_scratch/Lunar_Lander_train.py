#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:56:59 2023

@author: michiundslavki
"""

import numpy as np
import os
import torch.nn as nn
from utils.lunar_lander import Lunar_Lander_2
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#create model and log directories
models_dir = "./models/DDPG"

log_dir = "./logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = Lunar_Lander_2()#gym.make("Pendulum-v1", render_mode="rgb_array")
env.reset()
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
TIMESTEPS = 5000
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=[64, 64])

model = DDPG("MlpPolicy", env, action_noise=action_noise,policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001, buffer_size=1000000, 
             learning_starts=100, batch_size=100, tau=0.05, gamma=0.99, tensorboard_log=log_dir) #change tau 0.005 0.01 0.05
LOG_description = "reward_Model_4_test" 
model.learn(total_timesteps=TIMESTEPS, tb_log_name=LOG_description, log_interval=10)
model.save(f"{models_dir}/{TIMESTEPS}_{LOG_description}")