import gymnasium as gym
from my_custom_env.franka_env_cfg import MyFrankaLiftEnvCfg
from my_custom_env.franka_env_cfg_test_device import MyFrankaLiftEnvCfg_test_device

gym.register(
    id="My-Custom-Env-v0",  
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  
    kwargs={
        "env_cfg_entry_point": MyFrankaLiftEnvCfg,  
    },
    disable_env_checker=True,
)

gym.register(
    id="My-Custom-Env-v1",  
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  
    kwargs={
        "env_cfg_entry_point": MyFrankaLiftEnvCfg_test_device,  
    },
    disable_env_checker=True,
)