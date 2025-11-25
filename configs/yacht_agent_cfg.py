# marine_docking/configs/yacht_agent_cfg.py

from dataclasses import dataclass
from isaaclab_tasks.marine_docking.agents.rsl_rl.yacht_ppo_cfg import YachtPPORunnerCfg

@dataclass
class YachtAgentCfg(YachtPPORunnerCfg):
    """Yacht docking PPO configuration"""
    pass
