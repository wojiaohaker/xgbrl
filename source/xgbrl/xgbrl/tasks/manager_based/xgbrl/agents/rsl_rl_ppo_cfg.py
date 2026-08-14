# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class XgbRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "xgb_rough"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5,     # 降低初始 std，避免动作爆炸
            std_type="scalar", # 改用 scalar 类型，直接学习 std
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,     # 降低价值损失系数
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_mini_batches=4,
        learning_rate=3.0e-4,    # 提高学习率（官方默认值）
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        entropy_coef=0.01,
        num_learning_epochs=5,   # 增加更新次数
    )


@configclass
class XgbFlatPPORunnerCfg(XgbRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "xgb_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]
