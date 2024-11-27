import gymnasium as gym
from my_custom_env.franka_env_cfg import MyFrankaLiftEnvCfg

gym.register(
    id="My-Custom-Env-v0",  
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  
    kwargs={
        "env_cfg_entry_point": MyFrankaLiftEnvCfg,  
    },
    disable_env_checker=True,
)