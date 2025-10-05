from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import sys 
import numpy as np
from colorama import Fore 
import time

env = StableBaselinesGodotEnv(
    env_path='environments/2MovingPlatsEnemiesANDBigSlowPlat/Curriculum',
    # port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)

env = VecMonitor(env)
model = PPO.load('model_curriculum/6. NoFrameStack BigSlowPlat with TimeLimit/noframestack_curriculum_game_2_plats_WITH_enemies_AND_BigSlowPlat_10000000_steps', env=env)

print(evaluate_policy(model, env, deterministic=False, n_eval_episodes=10))