import gym
import os
import numpy as np
import torch

from breakout import DQNBreakout
from model import AtariNet
from agent import ReplayMemory,Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Konfigurasi perangkat
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
memory = ReplayMemory(capacity=1000, device=device)
# Inisialisasi
env = DQNBreakout(device=device, render_mode='human')
model = AtariNet(4)
model.to(device)
model.load_model()
agent = Agent(model=model,device=device,
              epsilon=1.0,nb_warmup=5e4,nb_actions=4,lr=1e-5,
              memory_capacity=1e6,batch_size=64)

agent.train(env=env,epcohs=200_000)