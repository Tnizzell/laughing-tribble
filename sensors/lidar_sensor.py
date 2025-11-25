from isaaclab.sensors import SensorBase
import torch

class YachtLidar(SensorBase):
    def __init__(self, num_beams=64, max_range=40.0, noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.max_range = max_range
        self.noise_std = noise_std

    def reset(self, env_ids=None):
        pass

    def get_obs(self):
        num_envs = self.num_envs
        data = torch.full((num_envs, self.num_beams), self.max_range, device=self.device)

        if self.noise_std > 0:
            data += torch.randn_like(data) * self.noise_std

        return data
