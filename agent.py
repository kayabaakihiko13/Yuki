import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import torch.optim as optim

from model import AtariNet
# from plot import LivePlot
class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []  # Circular buffer
        self.position = 0
        self.device = device

    def insert(self, transition):
        # Konversi semua elemen ke CPU sebelum menyimpan
        transition = [item.to('cpu') for item in transition]
        if len(self.memory)<self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            
            self.memory.append(transition)
        # self.position = (self.position + 1) % self.capacity  # Perbarui posisi secara melingkar

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size), "Not enough samples to draw from memory"
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10 

    def __len__(self):
        # Hitung jumlah elemen aktual dalam memori
        return len(self.memory)




class Agent:
    def __init__(self,model,device="cpu",epsilon=1.0,min_epsilon=1e-1,nb_warmup=1e4,nb_actions=None,
                 memory_capacity=1e4,batch_size:int=32,lr=25*1e-5):
        self.memory = ReplayMemory(device=device,capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon-min_epsilon)/nb_warmup)*2)
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions
        self.optimizer = optim.Adam(model.parameters(),lr=lr)
        print(f"starting epsilon is {self.epsilon}\n epsilon decay is {self.epsilon_decay}")
    
    def get_action(self,state):
        if torch.rand(1) <self.epsilon:
            return torch.randint(self.nb_actions,(1,1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av,dim=1,keepdim=True)
        
    
    def train(self,env,epcohs:int=10000):
        stats = {"Return":[],"AvgReturns":[],"EpsilonCheckPoint":[]}
        # plotter = LivePlot()
        for epoch in range(1,epcohs+1):
            state = env.reset()
            done = False
            ep_return = 0
            while not done:
                action = self.get_action(state)
                next_state,reward,done,_ = env.step(action)
                self.memory.insert([state,action,reward,done,next_state])
                if self.memory.can_sample(self.batch_size):
                    states, actions, rewards, dones, next_states = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        target_q_values = self.target_model(next_states)
                        max_next_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                    # Compute the expected Q-values using the model's predictions
                    current_q_values = self.model(states).gather(1, actions)
                    # Compute the expected Q-values
                    expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
                    # Compute the loss (Mean Squared Error loss)
                    loss = F.mse_loss(current_q_values, expected_q_values)
                    # update model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if epcohs % 1e4==0:
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay
            stats["Return"].append(ep_return)

            # Optionally calculate the average return
            avg_return = sum(stats["Return"]) / len(stats["Return"])
            stats["AvgReturns"].append(avg_return)
            stats["EpsilonCheckPoint"].append(self.epsilon)

            # Plot stats at each epoch (LivePlot should handle live plotting)
            # plotter.plot(stats["Return"], stats["AvgReturns"], stats["EpsilonCheckPoint"])

            print(f"Epoch {epoch}: Return {ep_return}, AvgReturn {avg_return}, Epsilon {self.epsilon}")

        return stats  # Return stats for further analysis or plotting
    def test(self,env):
        for epoch in range(1,3):
            state = env.reset()
            done = False
            for _ in range(10000):
                time.sleep(0.01)
                action = self.get_action(state)
                state,reward,done,_,info=env.step(action)
                if done:
                    break