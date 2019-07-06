import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

'''
Contains DQN and CNN-DQN classes
'''
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.num_actions = num_actions

    def forward(self, x, return_latent = 'second_hidden'):
        """Args:
        	 return_latent: 'last': return last hidden vector
           								'state': return the state
        """
        hidden1 = self.fc1(x)
        hidden2 = self.fc2(F.relu(hidden1))
        out = self.fc3(F.relu(hidden2))

        if return_latent == 'state':
        	return out, x 

        elif return_latent == 'first_hidden':
        	return out, hidden1 

        elif return_latent == 'second_hidden':
        	return out, hidden2 

        else:
        	print("Unrecognized return_latent argument: {}".format(return_latent))


    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value,_  = self.forward(state)
            action  = q_value.max(1)[1].data[0]
            action = int(action)
        else:
            action = random.randrange(self.num_actions)
        return action

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


# class CnnDQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(CnnDQN, self).__init__()
        
#         self.input_shape = input_shape
#         self.num_actions = num_actions
        
#         self.features = nn.Sequential(
#             nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#             #Flatten(),
#             #nn.Linear(self.feature_size(), 512),
#             #nn.ReLU(),
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(512, self.num_actions)
#         )
        

#     def forward(self, x, return_latent = 'last'):
#         print(x.shape)
#         hidden = self.features(x)
#         print(hidden.shape)
#         out = self.fc(out)
#         if return_latent == "state":
#             return out, x
#         return out, hidden
    
#     # H, W, 3
#     def feature_size(self):
#         return np.prod(self.input_shape)
#         #return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
#     def act(self, state, epsilon):
#         if random.random() > epsilon:
#             state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
#             q_value = self.forward(state)
#             action  = q_value.max(1)[1].data[0]
#         else:
#             action = random.randrange(self.num_actions)
#         return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
