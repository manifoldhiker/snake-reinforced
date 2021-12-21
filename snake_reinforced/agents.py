import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical, Normal, MultivariateNormal
from .game import preprocess_to_flatten_gray, ACTION2ID, ID2ACTION


class RandomAgent:
    def __init__(self, action_set=list(ACTION2ID.keys())):
        self.action_set = action_set

    def act(self, observation):
        action = np.random.choice(self.action_set)
        return {'action': action}


def identity(x): return x
def to_action_name(action): return ID2ACTION[action.item()]


class DefaultActorCritic(nn.Module):
    def __init__(self,
                 observation_space_size: int = 64*64,
                 action_space_size: int = len(ACTION2ID),
                 hidden_size: int = 256,
                 preprocess_fn=preprocess_to_flatten_gray,
                 postprocess_action_fn=identity,
                 device='cpu',
                 ):
        super().__init__()

        self.preprocess_fn = preprocess_fn
        self.postprocess_action_fn = postprocess_action_fn

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size,
                      out_features=hidden_size, bias=True),
            nn.PReLU(),

        )

        self.actor = nn.Linear(in_features=hidden_size,
                               out_features=action_space_size, bias=True)
        self.critic = nn.Linear(in_features=hidden_size,
                                out_features=1, bias=True)

        self.to(device)

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x

    def act(self, observation):
        x = self.preprocess_fn(observation)
        x = x.to(self.device)
        x = self(x)
        action_logits = self.actor(x)
        action = Categorical(logits=action_logits).sample()
        action = self.postprocess_action_fn(action)

        value = self.critic(x)
        return {'action_logits': action_logits, 'action': action, 'value': value}


class ActorCritic(DefaultActorCritic):
    def __init__(self,
                 observation_space_size: int = 64*64,
                 action_space_size: int = len(ACTION2ID),
                 hidden_size: int = 256,
                 preprocess_fn=preprocess_to_flatten_gray,
                 postprocess_action_fn=identity,
                 device='cpu',
                 ):

        super().__init__(observation_space_size,
                         action_space_size,
                         hidden_size,
                         preprocess_fn,
                         postprocess_action_fn,
                         device)

        self.preprocess_fn = preprocess_fn
        self.postprocess_action_fn = postprocess_action_fn

        self.device = device

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

        self.to(device)

    def forward(self, x):
        x = self.net(x)
        return x
