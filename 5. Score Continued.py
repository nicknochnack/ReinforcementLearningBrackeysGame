from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

import sys 
import numpy as np
from colorama import Fore 

env = StableBaselinesGodotEnv(
    port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=4) 

# model = PPO.load('model/brackey_200000_steps', env=env)
model = PPO.load('model/brackey_moving_plat_118000_steps')


episodes = range(10)
for episode in episodes: 
    print(Fore.LIGHTBLUE_EX + "\n\n\n⚡️ I'm at the pre-reset" + Fore.RESET)
    obs = env.reset()
    done = False
    score = 0 
    print(Fore.LIGHTYELLOW_EX +  f"🚀 Starting step loop {done}"+ Fore.RESET)
    while not done: 
        action, _state = model.predict(obs, deterministic=False)
        print(Fore.LIGHTYELLOW_EX +f"🏋🏽‍♀️ Actions selected: {action}, step about to be taken"+ Fore.RESET)
        obs, reward, done, info = env.step(action)
        print(Fore.LIGHTGREEN_EX + f"🤖 Step taken: {reward,done}"+ Fore.RESET)
        score+=reward
        print(Fore.LIGHTMAGENTA_EX + '🏆 Reward:{} Score:{}'.format(episode, score, done)+ Fore.RESET)
    print(Fore.LIGHTRED_EX + 'Episode DONE \n\n\n' + Fore.RESET) 
print("Attempting to close")
env.close()
print("Closed")
