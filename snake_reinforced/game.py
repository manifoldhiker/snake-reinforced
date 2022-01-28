import numpy as np
import torch

from ple.games.snake import Snake
from ple import PLE
from pygame.constants import K_w, K_a, K_s, K_d


def init_ple_env(width=64, height=64, init_length=3):
    game = Snake(width=width, height=height, init_length=init_length)
    ple_env = PLE(game, fps=30, display_screen=True)
    ple_env.init()
    return ple_env


ACTION2PLE_CODE = {
    "up": K_w,
    "left": K_a,
    "right": K_d,
    "down": K_s
}

ACTION2ID = {a: i for i, a in enumerate(["up", "left", "right", "down"])}
ID2ACTION = {i: a for a, i in ACTION2ID.items()}
PLE_CODE2ACTION = {v: k for k, v in ACTION2PLE_CODE.items()}


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def rgb_array2gray_tensor(rgb):
    return torch.tensor(rgb2gray(np.asarray(rgb)), dtype=torch.float32)[None, None, :]


def preprocess_to_flatten_gray(rgb):
    gray = rgb_array2gray_tensor(rgb)
    return gray.flatten()[None, :]


def get_snake_len(ple_env): return len(
    ple_env.getGameState()['snake_body_pos'])


class RandomAgent:
    def __init__(self, action_set=list(ACTION2ID.keys())):
        self.action_set = action_set

    def act(self, observation):
        action = np.random.choice(self.action_set)
        return {'action': action}


def play_game(agent, ple_env, max_frames=None):
    ple_env.reset_game()

    i = 0
    while True:
        i += 1
        if max_frames and i >= max_frames:
            break

        if ple_env.game_over():
            break

        observation = ple_env.getScreenRGB()
        act_result = agent.act(observation)
        action = act_result['action']

        if type(action) is torch.Tensor:
            action = ID2ACTION[action.item()]

        action_code = ACTION2PLE_CODE[action]
        reward = ple_env.act(action_code)

        yield {'observation': observation,
               'action': action,
               'reward': reward,
               'snake_len': get_snake_len(ple_env)
               }
