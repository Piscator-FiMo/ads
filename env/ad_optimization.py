from math import exp
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import OneHot, Composite, CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec #, BoundedContinuous
from torchrl.data.tensor_specs import Box, ContinuousBox, Unbounded, UnboundedContinuous
from torchrl.envs import EnvBase, make_composite_from_td


feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate", "cost_per_click"]

# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset: pd.DataFrame, initial_budget: int):
        super().__init__()
        self.dataset = dataset
        # batch_size ist hier die Environment Batch = Anzahl Environment, welche trainiert werden sollen.
        super().__init__(batch_size=torch.Size([]))  # ✅ Single environment

        self.feature_columns = feature_columns
        self.dataset = dataset
        self.num_features = len(feature_columns) + 1 #+1 for budget
        self.steps = 0
        self.initial_budget = initial_budget

        self.action_spec = Composite(action=OneHot(n=2, dtype=torch.int64))
        self.observation_spec = Composite(observation=Unbounded(shape=(self.num_features,), dtype=torch.float32))
        self.reward_spec = Composite(reward=Unbounded(shape=(1,), dtype=torch.float32))  # ✅ Corrected

        #self._reset(TensorDict({"done": torch.tensor(False)}))


    def _reset(self, tensordict=None):
        """Reset environment and return initial state."""
        sample = self.dataset.iloc[0]
        self.steps = 0
        self.budget = self.initial_budget

        state = torch.cat((torch.tensor(sample[feature_columns].values.astype(np.float32), dtype=torch.float32), torch.tensor([self.budget], dtype=torch.float32)))
        return TensorDict({
            "observation": state,
            #"budget": self.budget,
            "done": torch.tensor([False], dtype=torch.bool),  # Explicit shape [1]
        }, batch_size=[])

    def _step(self, tensordict):
        """Performs one step and returns the next state, reward, and done."""
        action = tensordict["action"].argmax().item()
        next_step = self.steps + 1
        next_sample = self.dataset.iloc[next_step]

        self.steps = self.steps + 1

        next_cost = next_sample["ad_spend"].item()

        if self.budget <= next_cost:
            action = 0  # when there is no more money, we can't buy ads

        if action == 1:
            self.budget -= next_cost - (next_sample["ad_conversions"].item() * 0.1)  # todo klären macht das sinn?

        next_state = torch.cat(
            (
                torch.tensor(next_sample[feature_columns].values.astype(np.float32), dtype=torch.float32),
                torch.tensor([self.budget], dtype=torch.float32),
            )
        )

        reward_value = self._compute_reward(action, next_sample)
        reward = torch.tensor(
            [reward_value], dtype=torch.float32
        )  # Shape [1], #reward und reward_spec müssen das gleiche Shape haben!

        done = next_step >= 365  # assumption we want to spend the budget over a year

        return TensorDict(
            {
                "observation": next_state,
                # "budget": torch.tensor(self.budget, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                #    "action": action
            }
        )  # ✅ Ensure batch size is correctly set

    def _compute_reward(self, action, sample) -> float:
        # reward prop zu roc
        #return self.compute_reward_of_row(action, sample["ad_roas"].values[0], sample["ad_spend"].values[0])
        return self.compute_reward_of_row(action, sample["ad_roas"], sample["ad_spend"])

    def compute_reward_of_row(self, action: int, ad_roas: float, ad_spent: float) -> float:
        return ad_roas if action == 1 else 0

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
