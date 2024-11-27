import gym
import os
import numpy as np
from PIL import Image
import torch

from breakout import *
from model import AtariNet
os.environ['KMP_DUPLICATE_LIB_OK'] ='TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

enviroment = DQNBreakout(device=device,render_mode='human')

model = AtariNet(nb_action=4)
model.to(device)
model.load_model()
state = enviroment.reset()

print(model.forward(state))
# for _ in range(100):
#     action = enviroment.action_space.sample()
#     state,reward,done,info = enviroment.step(action)
