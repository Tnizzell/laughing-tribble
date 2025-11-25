from isaaclab.sensors import SensorBase
import torch

class YachtRadar(SensorBase):
    def __init__(self, max_range=80.0, noise_std=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_range = max_range
        self.noise_std = noise_std

    def reset(self, env_ids=None):
        pass

    def get_obs(self):
        num_envs = self.num_envs
        obs = torch.zeros((num_envs, 2), device=self.device)

        if self.noise_std > 0:
            obs += torch.randn_like(obs) * self.noise_std

        return obs
