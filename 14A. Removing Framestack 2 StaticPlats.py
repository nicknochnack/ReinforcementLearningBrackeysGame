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
    env_path='environments/2PlatsNoEnemiesUnlimPlatReward/Curriculum',
    show_window=True, 
    n_parallel=10,
    speedup=32

)

env = VecMonitor(env)
model = PPO(
    "MultiInputPolicy",
    env, 
    ent_coef=0.0001,           # Higher entropy for exploration
    learning_rate=3e-4,     # Standard rate
    n_steps=42,           # Longer rollouts for timing patterns
    verbose=2,
    tensorboard_log='logs',
)

tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
            save_freq=(10000 // env.num_envs),
            save_path='model_curriculum',
            name_prefix='0. 2 NonMoving Plats No Enemies ULReward/noframestack_curriculum_game_2_plats_no_enemies',
        )


learn_arguments = dict(total_timesteps=1000000, tb_log_name='0. 2 NonMoving Plats No Enemies ULReward/noframestack_curriculum_game_2_plats_no_enemies', callback=checkpoint_callback)
model.learn(**learn_arguments)
export_model_as_onnx(model, os.path.join('model_curriculum/0. 2 NonMoving Plats No Enemies ULReward/noframestack_curriculum_game_2_plats_no_enemies.onnx'))

