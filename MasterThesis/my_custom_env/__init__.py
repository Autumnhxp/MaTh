import gymnasium as gym
from my_custom_env.env_cfg import MyCustomEnvCfg

gym.register(
    id="My-Custom-Env-v0",  
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  
    kwargs={
        "env_cfg_entry_point": MyCustomEnvCfg,  
    },
    disable_env_checker=True,
)