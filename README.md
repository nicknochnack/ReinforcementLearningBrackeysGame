
<!-- Sample Run -->
uv run godot_rl_agents/examples/stable_baselines3_example.py --env_path=examples/bin/godot_rl_JumperHard/bin/JumperHard.x86_64 --experiment_name=Experiment_01 --viz


Add AIController3D to player 
Right click AIController3D node and select extend script
Inside of here you'll handle get_obs, get_reward, get_action_space and set_action functions 