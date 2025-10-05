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
    # env_path='RingPong-ringpong_starter', 
    port=11008, 
    show_window=True, 
    n_parallel=1, # requires env path to be set if > 1  
    speedup=1
)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=4) 

# model = PPO(
#     "MultiInputPolicy",
#     env, 
#     ent_coef=0.1,           # Higher entropy for exploration
#     learning_rate=3e-4,     # Standard rate
#     n_steps=2048,           # Longer rollouts for timing patterns
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,             # Keep high for long-term rewards
#     gae_lambda=0.95,
#     clip_range=0.2,
#     verbose=2,
#     tensorboard_log='logs',
# )
model = PPO(
    "MultiInputPolicy",
    env, 
    ent_coef=0.01,           # Higher entropy for exploration
    learning_rate=3e-4,     # Standard rate
    n_steps=256,           # Longer rollouts for timing patterns
    verbose=2,
    tensorboard_log='logs',
)
tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(20000 // env.num_envs),
            save_path='model',
            name_prefix='brackey_better_rays',
        )

learn_arguments = dict(total_timesteps=400000, tb_log_name='brackey_better_rays', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('brackey_better_rays.onnx'))
