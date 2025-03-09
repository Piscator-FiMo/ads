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
    start_date = "2024-01-01"
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],                         # Suchanfragen zum bestimmten Keyword
        "timestamp": pd.date_range(start=start_date, periods=num_samples, freq="D"),     # Zeitstempel, täglich
        "impressions_rel": rand.randint(40, 100, num_samples),                           # Suchanfragen nach Keyword, relative Werte von 0 bis 100
        "competitiveness": rand.uniform(0, 1, num_samples),                              # Wettbewerbsintensität, 0 = wenig Konkurrenz, 1 = viel Konkurrenz, 
        "difficulty_score": rand.uniform(0, 1, num_samples),                             # Schwierigkeitsgrad für SEO-Rankings (0 = einfach, 1 = sehr schwer).
        "organic_rank": rand.randint(1, 11, num_samples),                                # Position in der organischen Suche (1 = ganz oben, 10 = unterste Position auf der ersten Seite).
        "organic_clicks": rand.randint(50, 5000, num_samples),                           # Anzahl der Klicks aus organischer Suche.
        "organic_ctr": rand.uniform(0.01, 0.3, num_samples),                             # Click-Through-Rate (CTR) der organischen Ergebnisse (Anteil der Nutzer, die auf das organische Ergebnis klicken).
        "paid_clicks": rand.randint(10, 3000, num_samples),                              # Anzahl der Klicks auf die bezahlte Anzeige.
        "paid_ctr": rand.uniform(0.01, 0.25, num_samples),                               # Click-Through-Rate der Anzeige (Wie oft die Anzeige geklickt wird, wenn sie erscheint).
        "ad_spend": rand.uniform(10, 10000, num_samples),                                # Werbeausgaben für das Keyword (in CHF).
        "ad_conversions": rand.uniform(0, 500, num_samples),                             # Anzahl der Conversions durch bezahlte Anzeigen (z. B. Käufe, Anmeldungen).
        "ad_roas": rand.uniform(0.5, 5, num_samples),                                    # Return on Ad Spend = conversion_value / ad_spend (Wie profitabel die Werbung ist).
        "conversion_rate": rand.uniform(0.01, 0.3, num_samples),                         # Anteil der Besucher, die eine gewünschte Aktion ausführen (z. B. Kauf).
        "cost_per_click": rand.uniform(0.1, 10, num_samples),                            # Kosten pro Klick (CPC) = ad_spend / paid_clicks.
        "cost_per_acquisition": rand.uniform(5, 500, num_samples),                       # Kosten pro Neukunde (CPA) = ad_spend / ad_conversions.
        "previous_recommendation": rand.choice([0, 1], size=num_samples),                # Wurde das Keyword zuvor für Werbung empfohlen? (0 = Nein, 1 = Ja).
        "impression_share": rand.uniform(0.1, 1.0, num_samples),                         # Marktanteil der Anzeigen (0 = kaum sichtbar, 1 = dominiert den Markt).
        "conversion_value": rand.uniform(0, 10000, num_samples)                          # Gesamtwert der erzielten Conversions in Währungseinheiten.
    }
    df = pd.DataFrame(data)

    #Abhängigkeit zwischen den Variablen einfügen, allerdings sollen die Daten weiterhin variieren, deshalb bleibt ein random-Teil enthalten.
    df["competitiveness"] = df["impressions_rel"] / 100 * rand.uniform(0.5,0.99)                         # Annahme: wenn oft nach einem Begriff gesucht wird, wird auch mehr Werbung geschaltet
    df["difficulty_score"] = df["competitiveness"] * rand.uniform(0.8,0.99)                              # Annahme: je höher die Konkurrenz ist, desto schwerer ist es die eigene Seite hoch im Ranking zu platzieren
    #organic
    df["organic_rank"] = np.clip((df["difficulty_score"] * rand.randint(1,11)).astype(int),1,10)         # Annahme: der Rang der eigenen Seite ist vom difficulty Score abhängig
    df["organic_clicks"] = df["organic_clicks"] * df["impressions_rel"] / 100 * rand.uniform(0.2, 0.5)   # Annahme: Anzahl Klicks ist abhängig von der Anzahl der Suchen    
    df["organic_ctr"] = df["organic_clicks"] / df["impressions_rel"]                                     # Annahme: CTR ist abhängig von der Anzahl der Suchen und den Anzahl Klicks
    #paid
    df["paid_clicks"] = df["paid_clicks"] * df["impressions_rel"] / 100 * rand.uniform(0.2, 0.5)         # Annahme: Anzahl Klicks ist abhängig von der Anzahl der Suchen    
    df["paid_ctr"] = df["paid_clicks"] / df["impressions_rel"]                                           # Annahme: CTR ist abhängig von der Anzahl der Suchen und den Anzahl Klicks
    df["ad_conversions"] = np.clip(df["ad_spend"] / 10000, 0.05, 1) * rand.uniform(0,500)                # Annahme: Die Conversion ist vom bezahlten Preis abhängig, np.clip = Wert muss zwischen 0.05 und 1 liegen
    df["conversion_value"] = df["ad_conversions"] * rand.uniform(10,150)                                 # Annahme: Pro Conversion wird ein Betrag zwischen 10 und 150 Franken ausgegeben. 
    #Berechnete KPI
    df["ad_roas"] = df["conversion_value"] / df["ad_spend"]                                              # wird mit Formel berechnet.
    df["cost_per_click"] = df["ad_spend"] / df["paid_clicks"]                                            # wird mit Formel berechnet.
    df["cost_per_acquisition"] = df["ad_spend"] / df["ad_conversions"]                                   # wird mit Formel berechnet.
    df["previous_recommendation"] = (data["ad_spend"] > 5000).astype(int)                                # Falls vorher viel ausgegeben wurde, empfehlen wir es erneut
    df["impression_share"] = np.clip(df["ad_spend"] / 10000, 0.05, 1) * rand.uniform(0.2,0.7)            # Annahme: Wenn eine Anzeige ein hohes Budget hat, wird die Anzeige auch oft ausgegeben
    return df

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














