from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

import sys 
import numpy as np

env = StableBaselinesGodotEnv(
    # env_path='RingPong-ringpong_starter', 
    port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)

for episode in range(10): 
    state = env.reset()
    num_envs = env.num_envs
    done = np.array(False).repeat(num_envs)  
    score = 0 

    while not all(done): 
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        n_state, reward, done, info = env.step(actions)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()