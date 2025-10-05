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

# model = PPO(
#     "MultiInputPolicy",
#     env, 
#     ent_coef=0.01,           # Higher entropy for exploration
#     learning_rate=1e-4,     # Standard rate
#     n_steps=256,           # Longer rollouts for timing patterns
#     verbose=2,
#     tensorboard_log='logs',
# )
model = PPO.load('model/ChangeRewardPlusFrameStackCont_1000000_steps', env=env)

tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(1000 // env.num_envs),
            save_path='model',
            name_prefix='ChangeRewardPlusFrameStackContOneHitReward',
        )

learn_arguments = dict(total_timesteps=1000000, tb_log_name='ChangeRewardPlusFrameStackContOneHitReward', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('ChangeRewardPlusFrameStackContOneHitReward.onnx'))

