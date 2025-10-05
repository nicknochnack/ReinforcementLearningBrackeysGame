from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure
from typing import Callable

from gymnasium.wrappers import TimeLimit
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

import sys 
import numpy as np
import os

env = StableBaselinesGodotEnv(
    port=11008,     
    env_path='environments/BeastLevelNoDeathPenalty/Curriculum',
    show_window=True, 
    n_parallel=10,
    speedup=32

)
env = TimeLimit(env, max_episode_steps=60)
env = VecMonitor(env)
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

model = PPO(
    "MultiInputPolicy",
    env, 
    ent_coef=0.001,           # Higher entropy for exploration
    learning_rate=linear_schedule(0.001),     # Standard rate
    n_steps=256,           # Longer rollouts for timing patterns
    verbose=2,
    tensorboard_log='logs',
)
tmp_path = "/tmp/sb3_log/BeastLevelNoDeathPenalty/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(100000 // env.num_envs),
            save_path='model_curriculum',
            name_prefix='9. BeastLevelRandoBeginning/BeastLevel',
        )



learn_arguments = dict(total_timesteps=10000000, tb_log_name='9. BeastLevelRandoBeginning/BeastLevel', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('model_curriculum/9. BeastLevelRandoBeginning/BeastLevel.onnx'))