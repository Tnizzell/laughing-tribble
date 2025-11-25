import gymnasium as gym

# Register the Marine Docking environment
gym.register(
    id="Marine-Docking-Yacht-v0",
    entry_point="isaaclab_tasks.marine_docking.envs.yacht_docking_env:YachtDockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.marine_docking.configs.yacht_env_cfg:YachtEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.marine_docking.configs.yacht_agent_cfg:YachtAgentCfg",
    },
)
