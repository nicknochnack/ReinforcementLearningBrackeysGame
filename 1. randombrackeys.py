from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

import time
import sys 
import numpy as np
from colorama import Fore

env = StableBaselinesGodotEnv(
    # env_path='RingPong-ringpong_starter', 
    port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=10) 


print(env.reset()) 

# for episode in range(1): 
#     print(Fore.LIGHTBLUE_EX + "\n\n\n⚡️ I'm at the pre-reset" + Fore.RESET)
#     state = env.reset()
#     num_envs = env.num_envs
#     done = [False]
#     score = 0 
#     print(Fore.LIGHTYELLOW_EX +  f"🚀 Starting step loop {done} {all(done)}"+ Fore.RESET)
#     while not all(done): 
#         actions = np.array([env.action_space.sample() for _ in range(num_envs)])
#         print(Fore.LIGHTYELLOW_EX +f"🏋🏽‍♀️ Actions selected: {actions}, step about to be taken"+ Fore.RESET)
#         n_state, reward, done, info = env.step(actions)
#         print(Fore.LIGHTGREEN_EX + f"🤖 Step taken: {reward,done}"+ Fore.RESET)
#         print(n_state) 
#         time.sleep(3)
#         score+=reward
#         print(Fore.LIGHTMAGENTA_EX + '🏆 Reward:{} Score:{}'.format(episode, score, done)+ Fore.RESET)
#     print(Fore.LIGHTRED_EX + 'Episode DONE \n\n\n' + Fore.RESET) 
# print("Attempting to close")
# env.close()
# print("Closed")
