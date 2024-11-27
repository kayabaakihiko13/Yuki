import collections
import cv2
import gym
import numpy as np
import torch
from PIL import Image

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat: int = 4, device='cpu'):
        env = gym.make('ALE/Breakout-v5', render_mode=render_mode)
        super(DQNBreakout, self).__init__(env)
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.frame_buffer = collections.deque(maxlen=2)  # Batasi buffer frame
        self.device = device
        self.image_resize = (84, 84)

    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            print(info, total_reward)

            # Kurangi skor jika kehilangan nyawa
            current_live = info['lives']
            if current_live < self.lives:
                total_reward -= 1
                self.lives = current_live

            self.frame_buffer.append(obs)
            if done:
                break

        # Ambil frame maksimum dari buffer terakhir
        max_frame = np.max(np.stack(self.frame_buffer), axis=0)
        max_frame = self.process_observation(max_frame)

        # Konversi total_reward dan done ke tensor PyTorch
        total_reward = torch.tensor([total_reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        return max_frame, total_reward, done, info
    def reset(self):
        self.frame_buffer = []
        obs,_ = self.env.reset()
        self.lives = self.env.ale.lives()
        obs = self.process_observation(observation=obs)
        return obs


    def process_observation(self, observation):
        # Proses observasi: skala abu-abu, ubah ukuran, normalisasi
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, self.image_resize, interpolation=cv2.INTER_AREA)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        observation /= 255.0  # Normalisasi ke [0, 1]
        return observation
