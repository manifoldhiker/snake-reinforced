import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical, Normal, MultivariateNormal
from .game import preprocess_to_flatten_gray, ACTION2ID, ID2ACTION


class RandomAgent:
    def __init__(self, ple_env):
        self.action_set = ple_env.getActionSet()

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
                 ):
        super().__init__()

        self.preprocess_fn = preprocess_fn
        self.postprocess_action_fn = postprocess_action_fn

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size,
                      out_features=hidden_size, bias=True),
            nn.PReLU(),

        )

        self.actor = nn.Linear(in_features=hidden_size,
                               out_features=action_space_size, bias=True)
        self.critic = nn.Linear(in_features=hidden_size,
                                out_features=1, bias=True)

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x

    def act(self, observation):
        x = self.preprocess_fn(observation)
        x = self(x)
        action_logits = self.actor(x)
        action = Categorical(logits=action_logits).sample()
        action = self.postprocess_action_fn(action)

        value = self.critic(x)
        return {'action_logits': action_logits, 'action': action, 'value': value}
