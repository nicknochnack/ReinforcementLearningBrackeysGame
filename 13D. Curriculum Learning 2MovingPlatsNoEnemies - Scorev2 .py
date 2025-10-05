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

env = StableBaselinesGodotEnv(
    env_path='environments/2MovingPlatsNoEnemies/Curriculum',
    # port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)

env = VecMonitor(env)
env = VecFrameStack(env, n_stack=25) 
model = PPO.load('model_curriculum/2. 2 Moving Plats No Enemies/curriculum_game_2_moving_plats_no_enemies_continued_2100000_steps', env=env)

episodes = range(10)
for episode in episodes: 
    print(Fore.LIGHTBLUE_EX + "\n\n\n⚡️ I'm at the pre-reset" + Fore.RESET)
    obs = None
    # obs = env.reset() - it doesn't like the reset for some reason? 
    done = [False]
    score = 0 
    print(Fore.LIGHTYELLOW_EX +  f"🚀 Starting step loop {done}"+ Fore.RESET)

    while not done[0]:
        if obs: 
            action, _state = model.predict(obs, deterministic=False)
        else: 
            action = np.array([[0,0]])
        print(Fore.LIGHTYELLOW_EX +f"🏋🏽‍♀️ Actions selected: {action}, step about to be taken"+ Fore.RESET)
        obs, reward, done, info = env.step(action)
        print(Fore.LIGHTGREEN_EX + f"🤖 Step taken: {reward,done}"+ Fore.RESET)
        score+=reward
        print(Fore.LIGHTMAGENTA_EX + '🏆 Reward:{} Score:{}'.format(episode, score, done)+ Fore.RESET)
    print(Fore.LIGHTRED_EX + 'Episode DONE \n\n\n' + Fore.RESET) 
print("Attempting to close")
env.close()
print("Closed")
