
import gymnasium as gym
import numpy as np
from gymnasium.utils.play import PlayPlot, play
import torch


def callback(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    print("info: ", info)
    return [reward,action]

plotter = PlayPlot(callback, horizon_timesteps=100, plot_names=["Immediate Rew.","Action"])

def play_continuous_mode():
    play(gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array", ), keys_to_action={
        "w": np.array([0, 0.7, 0]), 
        "a": np.array([-1, 0, 0]), "b": np.array([0, 0, 1]), 
        "d": np.array([1, 0, 0]), "wa": np.array([-1, 0.7, 0]),
        "dw": np.array([1, 0.7, 0]),"db": np.array([1, 0, 1]),
        "ab": np.array([-1, 0, 1]),},
        noop=np.array([0,0,0]))
 

def play_discrete_mode():
    play(gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array", ), keys_to_action={
        "d": 1, # right
        "a": 2, # left
        "w": 3, # gas
        "m": 4, # brake
        },
        noop=0,
    )

# to play the game, run `python keyboard_play.py` in the terminal of current directory 
# you can change the game mode by commenting out the mode you don't want to play

# play_continuous_mode()
play_discrete_mode()