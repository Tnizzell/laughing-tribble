from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.marine_docking.envs.yacht_docking_env import YachtDockingEnv
from isaaclab_tasks.marine_docking.configs.yacht_env_cfg import YachtEnvCfg
from isaaclab_tasks.marine_docking.configs.yacht_agent_cfg import YachtAgentCfg

@hydra_task_config(
    task_name="Marine-Docking-Yacht-v0",
    agent_cfg_entry_point="isaaclab_tasks.marine_docking.configs.yacht_agent_cfg:YachtAgentCfg",
)
def main(env_cfg, agent_cfg):
    return YachtDockingEnv(cfg=env_cfg), agent_cfg
