from math import exp
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import OneHot, Composite, Unbounded
from torchrl.envs import EnvBase


# make it win and loose money by its decisions
# done if no more budget
# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset: pd.DataFrame, feature_columns: list[str], budget: float):
        super().__init__()
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.action_spec = Composite(action=OneHot(n=2, dtype=torch.int64))
        self.steps = 0
        self.budget = budget
        self.step_size = 1
        self.observation_spec = Composite(
            observation=Composite(data=Unbounded(shape=(self.step_size, self.num_features), dtype=torch.float32),
                                  budget=Unbounded(shape=torch.Size([1]), dtype=torch.float32)))
        self.reward_spec = Composite(
            reward=Unbounded(shape=torch.Size([1]), dtype=torch.float64)
        )
        self._reset(TensorDict({"done": torch.tensor(False)}))

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.steps = 0
        sample, _ = self._next_slice()
        state = torch.tensor(sample[self.feature_columns].values, dtype=torch.float32)
        return TensorDict({"observation": TensorDict({"data": state, "budget": torch.tensor([self.budget])})},
                          batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        next_sample, next_step = self._next_slice()
        if next_step > self.dataset.size:
            done = True
            return TensorDict({"done": torch.tensor(done)}, batch_size=[])
        # todo how are we going through by keyword would require multiple decisions
        budget = tensordict["observation"]["budget"][0].item()
        if action == 1:
            budget -= (next_sample[-1:]["ad_spend"].item() - next_sample[-1:]["ad_conversions"].item())
        next_state = torch.tensor(next_sample[self.feature_columns].values, dtype=torch.float32)
        reward = self._compute_reward(action, next_sample[-1:])
        done = budget < 0
        return TensorDict({
            "observation": TensorDict({"data": next_state, "budget": torch.tensor([budget])}),
            "reward": torch.tensor([np.float64(reward)]),
            "done": torch.tensor(done)
        }, batch_size=[])

    def _next_slice(self):
        next_step = self.steps + self.step_size
        next_sample = self.dataset[self.steps:next_step]
        self.steps = self.steps + 1
        return next_sample, next_step

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
