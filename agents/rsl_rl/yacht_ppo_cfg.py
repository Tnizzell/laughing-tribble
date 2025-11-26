# marine_docking/agents/rsl_rl/yacht_ppo_cfg.py

from dataclasses import dataclass, field
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@dataclass
class YachtPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = "marine_docking_yacht"

    # ---- FIX: MUST USE default_factory ----
    policy: RslRlPpoActorCriticCfg = field(
        default_factory=lambda: RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            activation="elu",
        )
    )

    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
