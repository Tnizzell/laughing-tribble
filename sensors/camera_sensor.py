from isaaclab.sensors import SensorBase
import torch

class YachtCamera(SensorBase):
    def __init__(self, feature_dim=32, noise_std=0.01, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.noise_std = noise_std

    def reset(self, env_ids=None):
        pass

    def get_obs(self):
        num_envs = self.num_envs
        obs = torch.zeros((num_envs, self.feature_dim), device=self.device)

        if self.noise_std > 0:
            obs += torch.randn_like(obs) * self.noise_std

        return obs
