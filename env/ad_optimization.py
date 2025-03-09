from math import exp
from typing import Optional

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import OneHot
from torchrl.envs import EnvBase


# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset: pd.DataFrame, feature_columns: list[str]):
        super().__init__()
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.action_spec = OneHot(n=2, dtype=torch.int64)
        self._reset()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        sample = self.dataset.sample(1)
        state = torch.tensor(sample[self.feature_columns].values, dtype=torch.float32).squeeze()
        return TensorDict({"observation": state}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        next_sample = self.dataset.sample(1)
        next_state = torch.tensor(next_sample[self.feature_columns].values, dtype=torch.float32).squeeze()
        reward = self._compute_reward(action, next_sample)
        done = False
        return TensorDict({"observation": next_state, "reward": torch.tensor(reward), "done": torch.tensor(done)}, batch_size=[])

    def _compute_reward(self, action, sample) -> float:
        # reward prop zu roc
        ad_roas = sample["ad_roas"]
        reward = 1 / (1 + exp(-ad_roas))
        if action == 1 and ad_roas > 5000:
            # reward a very good action a lot
            reward = ad_roas
        elif action == 1 and ad_roas < 0:
            # if the conversion value < ad spent penalise
            reward = -sample["ad_spent"] * reward
        elif action == 0 and ad_roas < 0:
            # reward proportionally to saved money
            reward = reward
        elif action == 0 and ad_roas > 0:
            # if not bought but would be profitable penalise
            reward = -reward
        return reward

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
