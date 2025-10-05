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
from colorama import Fore 

env = StableBaselinesGodotEnv(
    port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=4) 

model = PPO.load('model/brackey_32000_steps', env=env)
tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(10000 // env.num_envs),
            save_path='model',
            name_prefix='brackey_moving_plat_changed_ray_pos',
        )

learn_arguments = dict(total_timesteps=10000000, tb_log_name='brackey_moving_plat', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('brackey.onnx'))
