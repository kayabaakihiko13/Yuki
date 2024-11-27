import gym
import os
import numpy as np
from PIL import Image
import torch

from breakout import *
os.environ['KMP_DUPLICATE_LIB_OK'] ='TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

enviroment = DQNBreakout(device=device,render_mode='human')
state = enviroment.reset()
for _ in range(100):
    action = enviroment.action_space.sample()
    state,reward,done,info = enviroment.step(action)
    