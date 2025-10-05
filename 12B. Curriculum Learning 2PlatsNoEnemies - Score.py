from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3.common.env_util import make_vec_env

import sys 
import numpy as np
from colorama import Fore 
import time


episodes = range(3)
for episode in episodes: 
    env = StableBaselinesGodotEnv(
        env_path='environments/2PlatsNoEnemies/Curriculum',
        # port=11008, 
        show_window=True, 
        n_parallel=1, # requires env path to be set if > 1  
        speedup=1
    )
    
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=25) 
    model = PPO.load('model_curriculum/1. 2 NonMoving Plats No Enemies/curriculum_game_2_plats_no_enemies_70000_steps', env=env)
    print(Fore.LIGHTBLUE_EX + "\n\n\n⚡️ I'm at the pre-reset" + Fore.RESET)
    obs = None
    obs = env.reset()    
    done = [False]
    score = 0 
    print(Fore.LIGHTYELLOW_EX +  f"🚀 Starting step loop {done}"+ Fore.RESET)

    while not done[0]:
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
