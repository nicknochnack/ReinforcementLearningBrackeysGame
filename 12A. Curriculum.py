from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

import sys 
import numpy as np
import os

env = StableBaselinesGodotEnv(
    port=11008, 
    show_window=True, 
    n_parallel=1,
    speedup=1
)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=25) 

model = PPO(
    "MultiInputPolicy",
    env, 
    ent_coef=0.0001,           # Higher entropy for exploration
    learning_rate=3e-4,     # Standard rate
    n_steps=32,           # Longer rollouts for timing patterns
    verbose=2,
    tensorboard_log='logs',
)

tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(10000 // env.num_envs),
            save_path='model_curriculum',
            name_prefix='1. 2 NonMoving Plats No Enemies/curriculum_game_2_plats_no_enemies',
        )


learn_arguments = dict(total_timesteps=1000000, tb_log_name='1. 2 NonMoving Plats No Enemies/curriculum_game_2_plats_no_enemies', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('model_curriculum/1. 2 NonMoving Plats No Enemies/curriculum_game_2_plats_no_enemies.onnx'))

