from math import exp
from typing import Optional

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import OneHot, CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec #, BoundedContinuous
from torchrl.data.tensor_specs import Box, ContinuousBox, Unbounded, UnboundedContinuous
from torchrl.envs import EnvBase, make_composite_from_td


# Todo add a budget
# todo env specs
# make it win and loose money by its decisions
# done if no more budget
# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset: pd.DataFrame, feature_columns: list[str], budget: int):
        super().__init__()
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.action_spec = OneHot(n=2, dtype=torch.int64)
        self.steps = 0
        self.budget = budget
        self._reset(TensorDict({"done": torch.tensor(False)}))

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        sample = self.dataset.sample(1)
        self.steps = 0
        state = torch.tensor(sample[self.feature_columns].values, dtype=torch.float32)
        return TensorDict({"observation": TensorDict({"data": state, "budget": self.budget})}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        next_step = self.steps + 15
        next_sample = self.dataset[self.steps:next_step]
        # todo how are we going through by keyword would require multiple decisions
        self.steps = self.steps + 1
        budget = tensordict["observation"]["budget"].item()
        if action == 1:
            budget -=  (next_sample[-1:]["ad_spend"].item() - next_sample[-1:]["ad_conversions"].item())
        next_state = torch.tensor(next_sample[self.feature_columns].values, dtype=torch.float32)
        reward = self._compute_reward(action, next_sample[-1:])
        done = budget < 0
        return TensorDict(
            {"observation": TensorDict({"data": next_state, "budget": torch.tensor(budget)}), "reward": torch.tensor(reward),
             "done": torch.tensor(done)},
            batch_size=[])

    def _compute_reward(self, action, sample) -> float:
        # reward prop zu roc
        return self.compute_reward_of_row(action, sample["ad_roas"].values[0], sample["ad_spend"].values[0])

    def compute_reward_of_row(self, action: int, ad_roas: float, ad_spent: float) -> float:
        reward = 1 / (1 + exp(-ad_roas))
        if action == 1 and ad_roas > 5000:
            # reward a very good action a lot
            reward = ad_roas
        elif action == 1 and ad_roas < 1:
            # if the conversion value < ad spent penalise
            reward = -ad_spent * reward
        elif action == 0 and ad_roas < 1:
            # reward proportionally to saved money
            reward = reward
        elif action == 0 and ad_roas > 1:
            # if not bought but would be profitable penalise
            reward = -reward
        return reward

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
