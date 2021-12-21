import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from snake_reinforced.game import ACTION2ID, play_game
from snake_reinforced.infrastructure import ensure_parent_exists

MAX_ACTION_LEN = max([len(a) for a in ACTION2ID.keys()])


def animate_frames(frames):
    fig = plt.figure()
    ax = fig.gca()

    def animate(step, ax):
        ax.cla()
        frame = frames[step]
        title = f'step={step:04d}, snake_len={frame["snake_len"]:02d}'
        im = ax.imshow(frame['observation'], animated=True)
        im = ax.set_title(title)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), fargs=(ax,), interval=100)

    return anim


def save_snake_video(ple_env, agent, filename, n_last_frames=200):
    ensure_parent_exists(filename)

    frames = list(play_game(agent, ple_env))
    frames = frames[-n_last_frames:]
    anim = animate_frames(frames)
    anim.save(filename)
