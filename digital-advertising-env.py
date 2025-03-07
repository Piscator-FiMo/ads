!pip install torch==2.5.0 torchrl

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as rand
import random
import pandas as pd
import time

from typing import Optional

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase, TransformedEnv, StepCounter
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.data import OneHot, DiscreteTensorSpec, LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torch.optim import Adam

import matplotlib.pyplot as plt

# Generate Realistic Synthetic Data
def generate_synthetic_data(num_samples=1000):
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],
        "competitiveness": np.random.uniform(0, 1, num_samples),
        "difficulty_score": np.random.uniform(0, 1, num_samples),
        "organic_rank": np.random.uniform(1, 10, num_samples),
        "organic_clicks": np.random.randint(50, 5000, num_samples),
        "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),
        "paid_clicks": np.random.randint(10, 3000, num_samples),
        "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),
        "ad_spend": np.random.uniform(10, 10000, num_samples),
        "ad_conversions": np.random.uniform(0, 500, num_samples),
        "ad_roas": np.random.uniform(0.5, 5, num_samples),
        "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),
        "cost_per_click": np.random.uniform(0.1, 10, num_samples),
        "cost_per_acquisition": np.random.uniform(5, 500, num_samples),
        "previous_recommendation": np.random.choice([0, 1], size=num_samples),
        "impression_share": np.random.uniform(0.1, 1.0, num_samples),
        "conversion_value": np.random.uniform(0, 10000, num_samples)
    }
    return pd.DataFrame(data)

# Load synthetic dataset
dataset = generate_synthetic_data(1000)
feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr", "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate", "cost_per_click"]
dataset

# Define a Custom TorchRL Environment
class AdOptimizationEnv(EnvBase):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.num_features = len(feature_columns)
        self.action_spec = OneHot(n=2, dtype=torch.int64)
        self._reset()

    def _reset(self, tensordict=None):
        sample = self.dataset.sample(1)
        state = torch.tensor(sample[feature_columns].values, dtype=torch.float32).squeeze()
        return TensorDict({"observation": state}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        #action = tensordict["action"].item()
        next_sample = self.dataset.sample(1)
        next_state = torch.tensor(next_sample[feature_columns].values, dtype=torch.float32).squeeze()
        reward = self._compute_reward(action, next_sample)
        done = False
        return TensorDict({"observation": next_state, "reward": torch.tensor(reward), "done": torch.tensor(done)}, batch_size=[])

    def _compute_reward(self, action, sample):
        cost = sample["ad_spend"].values[0]
        ctr = sample["paid_ctr"].values[0]
        if action == 1 and cost > 5000:
            reward = 1.0
        elif action == 0 and ctr > 0.15:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

# Initialize Environment
env = AdOptimizationEnv(dataset)
env = TransformedEnv(env, StepCounter())
state_dim = env.num_features
action_dim = env.action_spec.n


env.action_spec

value_mlp = MLP(in_features=env.num_features, out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = TensorDictSequential(policy, exploration_module)

value_mlp

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)


total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10:
                print(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

print(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)














