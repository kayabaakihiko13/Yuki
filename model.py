import torch
import torch.nn as nn

class AtariNet(nn.Module):
    def __init__(self, nb_action=4):
        super(AtariNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        
        # Action-value branch
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_action)
        
        # State-value branch
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):  # Jika input bukan Tensor
            x = torch.Tensor(x).to(next(self.parameters()).device)  # Kirim ke perangkat model
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        
        # State-value branch
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.state_value3(state_value)  # Skalar (nilai keadaan)
        
        # Action-value branch
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.action_value3(action_value)  # Output tindakan
        
        # Combine state and action values
        q_values = state_value + (action_value - action_value.mean(dim=1, keepdim=True))
        return q_values
    def save_model(self,weight_filename:str='models/latest.pt'):
        torch.save(self.state_dict(),weight_filename)
    def load_model(self,weight_filename:str="models/latest.pt"):
        try:
            self.load_state_dict(torch.load(weight_filename))
            print(f"Successfully loaded weight file {weight_filename}")
        except:
            print(f"No weight file available at{weight_filename}")
