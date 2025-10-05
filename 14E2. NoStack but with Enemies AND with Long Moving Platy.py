from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure

from gymnasium.wrappers import TimeLimit
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

import sys 
import numpy as np
import os

env = StableBaselinesGodotEnv(
    port=11008,     
    env_path='environments/2MovingPlatsEnemiesANDBigSlowPlat/Curriculum',
    show_window=True, 
    n_parallel=10,
    speedup=32

)
env = TimeLimit(env, max_episode_steps=60)
env = VecMonitor(env)
model = PPO.load('model_curriculum/4. NoFrameStack 2 NonMoving Plats with Enemies/noframestack_curriculum_game_2_plats_WITH_enemies_1000000_steps', env=env)
model.ent_coef =0.001
tmp_path = "/tmp/sb3_log/2MovingPlatsNoFramestackWithEnemiesANDBigSlowPlat"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(100000 // env.num_envs),
            save_path='model_curriculum',
            name_prefix='6. NoFrameStack BigSlowPlat with TimeLimit/noframestack_curriculum_game_2_plats_WITH_enemies_AND_BigSlowPlat',
        )


learn_arguments = dict(total_timesteps=10000000, tb_log_name='6. NoFrameStack BigSlowPlat with TimeLimit/noframestack_curriculum_game_2_plats_WITH_enemies_AND_BigSlowPlat', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('model_curriculum/6. NoFrameStack BigSlowPlat with TimeLimit/noframestack_curriculum_game_2_plats_WITH_enemies_AND_BigSlowPlat.onnx'))

