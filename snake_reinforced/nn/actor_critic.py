import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical, Normal, MultivariateNormal
from snake_reinforced.game import preprocess_to_flatten_gray, rgb_array2gray_tensor, ACTION2ID, ID2ACTION


def identity(x): return x
def to_action_name(action): return ID2ACTION[action.item()]


class ActorCriticPerceptron(nn.Module):
    def __init__(self,
                 observation_space_size: int = 64*64,
                 action_space_size: int = len(ACTION2ID),
                 hidden_size: int = 256,):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(observation_space_size),
            nn.Linear(in_features=observation_space_size,
                      out_features=hidden_size, bias=True),
            nn.PReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(in_features=hidden_size,
                      out_features=hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),

            nn.Linear(in_features=hidden_size,
                      out_features=action_space_size, bias=True)
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features=hidden_size,
                      out_features=hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),

            nn.Linear(in_features=hidden_size,
                      out_features=1, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        h = self.net(x)

        action_logits = self.actor(h)
        values = self.critic(h)
        return action_logits, values


class ActorCriticConvNet(nn.Module):
    def __init__(self,
                 observation_space_size: int = 64*64,
                 action_space_size: int = len(ACTION2ID),
                 hidden_size: int = 128,):
        super().__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 8, 3, 1)

        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(8, 16, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.0)
        self.dropout2 = nn.Dropout2d(0.0)

        # First fully connected layer
        self.fc1 = nn.Linear(3136, hidden_size)

        self.actor = nn.Sequential(
            nn.Linear(in_features=hidden_size,
                      out_features=hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),

            nn.Linear(in_features=hidden_size,
                      out_features=action_space_size, bias=True)
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features=hidden_size,
                      out_features=hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),

            nn.Linear(in_features=hidden_size,
                      out_features=1, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        h = F.relu(x)

        action_logits = self.actor(h)
        values = self.critic(h)
        return action_logits, values


class ActorCriticAgent(nn.Module):
    def __init__(self,
                 actor_critic_net,
                 preprocess_fn=rgb_array2gray_tensor,
                 postprocess_action_fn=identity,
                 device='cpu',
                 ):
        super().__init__()

        self.preprocess_fn = preprocess_fn
        self.postprocess_action_fn = postprocess_action_fn

        self.device = device

        self.actor_critic_net = actor_critic_net

        self.to(device)

    def act(self, observation):
        x = self.preprocess_fn(observation)
        x = x.to(self.device)

        action_logits, value = self.actor_critic_net(x)
        action = Categorical(logits=action_logits).sample()
        action = self.postprocess_action_fn(action)

        return {'action_logits': action_logits, 'action': action, 'value': value}
