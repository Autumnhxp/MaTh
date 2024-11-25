from omni.isaac.lab.envs import ManagerBasedRLEnv
from .env_cfg import MyCustomEnvCfg  # 引用配置類

class MyCustomEnv(ManagerBasedRLEnv):
    def __init__(self, env_cfg):
        super().__init__(env_cfg)
        self.env_cfg = env_cfg

    def reset(self):
        obs = super().reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
