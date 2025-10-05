from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

import sys 
import numpy as np
import os

env = StableBaselinesGodotEnv(
    port=11008,     
    env_path='environments/2MovingPlatsEnemies/Curriculum',
    show_window=True, 
    n_parallel=10, # prev 10
    speedup=32 # prev 32

)

env = VecMonitor(env)
model = PPO.load('model_curriculum/3. NoFrameStack 2 NonMoving Plats No Enemies/noframestack_curriculum_game_2_plats_no_enemies_610000_steps', env=env)
tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(100000 // env.num_envs), # prev 10k
            save_path='model_curriculum',
            name_prefix='4. NoFrameStack 2 NonMoving Plats with Enemies/noframestack_curriculum_game_2_plats_WITH_enemies',
        )


learn_arguments = dict(total_timesteps=1000000, tb_log_name='4. NoFrameStack 2 NonMoving Plats with Enemies/noframestack_curriculum_game_2_plats_WITH_enemies', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('model_curriculum/4. NoFrameStack 2 NonMoving Plats with Enemies/noframestack_curriculum_game_2_plats_WITH_enemies.onnx'))

