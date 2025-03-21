#!pip install torch==2.5.0 torchrl
#!pip install torchrl==0.7.2

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

from env.ad_optimization import AdOptimizationEnv


# Generate Realistic Synthetic Data
def add_generated_synthetic_data(preprocessed_data: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    df: pd.DataFrame = preprocessed_data.copy()
    rnd = rand.RandomState(seed)

    df["timestamp"] = df["Day"].apply(lambda day: pd.to_datetime(day, format="%Y-%m-%d"))   # Zeitstempel, täglich
    df["keyword"] = df["keyword"]                                                           # Suchanfragen zum bestimmten Keyword
    df["impressions_rel"] = df["freq"]                                                      # Suchanfragen nach Keyword, relative Werte von 0 bis 100
    df["competitiveness"] = df.apply(lambda _: rnd.uniform(0, 1),axis=1)          # Wettbewerbsintensität, 0 = wenig Konkurrenz, 1 = viel Konkurrenz,
    df["difficulty_score"] = df.apply(lambda _: rnd.uniform(0, 1),axis=1)         # Schwierigkeitsgrad für SEO-Rankings (0 = einfach, 1 = sehr schwer).
    df["organic_rank"] = df.apply(lambda _: rnd.randint(1, 11), axis=1)           # Position in der organischen Suche (1 = ganz oben, 10 = unterste Position auf der ersten Seite).
    df["organic_clicks"] = df.apply(lambda _: rnd.randint(50, 5000), axis=1)      # Anzahl der Klicks aus organischer Suche.
    df["organic_ctr"] = df.apply(lambda _: rnd.uniform(0.01, 0.3),axis=1)         # Click-Through-Rate (CTR) der organischen Ergebnisse (Anteil der Nutzer, die auf das organische Ergebnis klicken).
    df["paid_clicks"] = df.apply(lambda _: rnd.randint(10, 3000), axis=1)         # Anzahl der Klicks auf die bezahlte Anzeige.
    df["paid_ctr"] = df.apply(lambda _: rnd.uniform(0.01, 0.25),axis=1)           # Click-Through-Rate der Anzeige (Wie oft die Anzeige geklickt wird, wenn sie erscheint).
    df["ad_spend"] = df.apply(lambda _: rnd.uniform(10, 10000), axis=1)           # Werbeausgaben für das Keyword (in CHF).
    df["ad_conversions"] = df.apply(lambda _: rnd.uniform(0, 500),axis=1)         # Anzahl der Conversions durch bezahlte Anzeigen (z. B. Käufe, Anmeldungen).
    df["ad_roas"] = df.apply(lambda _: rnd.uniform(0.5, 5),axis=1)                 # Return on Ad Spend = conversion_value / ad_spend (Wie profitabel die Werbung ist).
    df["conversion_rate"] = df.apply(lambda _: rnd.uniform(0.01, 0.3),axis=1)     # Anteil der Besucher, die eine gewünschte Aktion ausführen (z. B. Kauf).
    df["cost_per_click"] = df.apply(lambda _: rnd.uniform(0.1, 10),axis=1)        # Kosten pro Klick (CPC) = ad_spend / paid_clicks.
    df["cost_per_acquisition"] = df.apply(lambda _: rnd.uniform(5, 500),axis=1)   # Kosten pro Neukunde (CPA) = ad_spend / ad_conversions.
    df["previous_recommendation"] = df.apply(lambda _: rnd.choice([0, 1]),axis=1)            # Wurde das Keyword zuvor für Werbung empfohlen? (0 = Nein, 1 = Ja).
    df["impression_share"] = df.apply(lambda _: rnd.uniform(0.1, 1.0),axis=1)     # Marktanteil der Anzeigen (0 = kaum sichtbar, 1 = dominiert den Markt).
    df["conversion_value"] = df.apply(lambda _: rnd.uniform(0, 10000),axis=1)     # Gesamtwert der erzielten Conversions in Währungseinheiten.

    # Abhängigkeit zwischen den Variablen einfügen, allerdings sollen die Daten weiterhin variieren, deshalb bleibt ein random-Teil enthalten.
    df["competitiveness"] = df["impressions_rel"] / 100 * rnd.uniform(0.5,0.99)   # Annahme: wenn oft nach einem Begriff gesucht wird, wird auch mehr Werbung geschaltet
    df["difficulty_score"] = df["competitiveness"] * rnd.uniform(0.8,0.99)        # Annahme: je höher die Konkurrenz ist, desto schwerer ist es die eigene Seite hoch im Ranking zu platzieren
    # organic
    df["organic_rank"] = np.clip((df["difficulty_score"] * rnd.randint(1, 11)).astype(int), 1,10)  # Annahme: der Rang der eigenen Seite ist vom difficulty Score abhängig
    df["organic_clicks"] = df["organic_clicks"] * df["impressions_rel"] / 100 * rnd.uniform(0.2,0.5)            # Annahme: Anzahl Klicks ist abhängig von der Anzahl der Suchen
    df["organic_ctr"] = df["organic_clicks"] / df["impressions_rel"]                                                      # Annahme: CTR ist abhängig von der Anzahl der Suchen und den Anzahl Klicks
    # paid
    df["paid_clicks"] = df["paid_clicks"] * df["impressions_rel"] / 100 * rnd.uniform(0.2,0.5)         # Annahme: Anzahl Klicks ist abhängig von der Anzahl der Suchen
    df["paid_ctr"] = df["paid_clicks"] / df["impressions_rel"]                                                   # Annahme: CTR ist abhängig von der Anzahl der Suchen und den Anzahl Klicks
    df["ad_conversions"] = np.clip(df["ad_spend"] / 10000, 0.05, 1) * rnd.uniform(0,500)  # Annahme: Die Conversion ist vom bezahlten Preis abhängig, np.clip = Wert muss zwischen 0.05 und 1 liegen
    df["conversion_value"] = df["ad_conversions"] * rnd.uniform(100,1000)                                # Annahme: Pro Conversion wird ein Betrag zwischen 10 und 150 Franken ausgegeben.
    # Berechnete KPI
    df["ad_roas"] = df["conversion_value"] / df["ad_spend"]              # wird mit Formel berechnet.
    df["cost_per_click"] = df["ad_spend"] / df["paid_clicks"]            # wird mit Formel berechnet.
    df["cost_per_acquisition"] = df["ad_spend"] / df["ad_conversions"]   # wird mit Formel berechnet.
    df["previous_recommendation"] = (df["ad_spend"] > 5000).astype(int)  # Falls vorher viel ausgegeben wurde, empfehlen wir es erneut
    df["impression_share"] = np.clip(df["ad_spend"] / 10000, 0.05, 1) * rnd.uniform(0.2,0.7)  # Annahme: Wenn eine Anzeige ein hohes Budget hat, wird die Anzeige auch oft ausgegeben

    # Drop the original columns
    df = df.drop(columns=["Day", "freq"])

    # Reorder the columns
    df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]

    # Set the timestamp as the index
    df = df.set_index("timestamp", drop=False)

    # Print some information about the dataset
    print(df.head())
    df.info()
    print(df["ad_roas"].describe(include="all"))
    #plt.hist(df["ad_roas"])
    #plt.show()
    return df


def load_data() -> pd.DataFrame:
    return pd.read_csv("data/preprocessed.csv")


# Load synthetic dataset
dataset = add_generated_synthetic_data(preprocessed_data=load_data(), seed=42)
dataset = dataset[dataset["keyword"] =="iPhone"]

feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr",
                   "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate",
                   "cost_per_click"]

if __name__ == "__main__":

    # Initialize Environment
    env = AdOptimizationEnv(dataset, budget=1_000_000)
    env = TransformedEnv(env, StepCounter(max_steps=50_000))

    print("Observation Keys:", env.observation_keys)
    print("Action Keys:", env.action_keys)
    print("Reward Keys:", env.reward_keys)

    print("env.observation_spec:\r\n", env.observation_spec)
    print("env.action_spec:\r\n", env.action_spec)
    print("env.reward_spec:\r\n", env.reward_spec)

    #state_dim = env.num_features
    #action_dim = env.action_spec.n
    #print(env.observation_spec)
    #td = env.reset()
    #print(td)
    #td = env.rand_step(td)
    #print(td)

    value_mlp = MLP(in_features=env.num_features, out_features=env.action_spec.shape[-1], num_cells=[64, 64])
    value_net = TensorDictModule(value_mlp, in_keys=["observation"],
                                 out_keys=["action_value"])  # observation und action_value müssen genau so heissen
    policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))
    exploration_module = EGreedyModule(
        env.action_spec, annealing_num_steps=100_000, eps_init=0.9
        # annealing_num_steps darf nicht höher sein als total_frames
    )
    policy_explore = TensorDictSequential(policy, exploration_module)

    print("value_mlp: \r\n", value_mlp)
    print("policy: \r\n", policy)


    init_rand_steps = 500
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy_explore,
        frames_per_batch=1,  # ✅ Match the batch size
        total_frames=10_000,
        init_random_frames=init_rand_steps,
        storing_device="cpu",  # ✅ Ensure storing happens on CPU
        split_trajs=False,  # ✅ Prevent trajectory splitting from dropping keys
        exploration_type="mode"
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))



    loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
    optim = Adam(loss.parameters(), lr=0.02)
    updater = SoftUpdate(loss, eps=0.99)


    total_count = 0
    total_episodes = 0
    optim_steps = 10
    t0 = time.time()
    for i, data in enumerate(collector):
        # Write data in replay buffer
        print(i, ": data:", data["action_value"])
        if i == 47:
            print("i == ", i)

        rb.extend(data)
        max_length = rb[:]["next", "step_count"].max()
        print("max_length", max_length)
        if len(rb) > init_rand_steps:
            # per batch collected for efficiency)
            for _ in range(optim_steps):
                sample = rb.sample(128)
                loss_vals = loss(sample)
                print("loss function is called:", loss_vals["loss"])
                loss_vals["loss"].backward()
                optim.step()
                optim.zero_grad()
                # Update exploration factor
                exploration_module.step(data.numel())
                # Update target params
                updater.step()
                if i % 100:
                    print(f"Max num steps: {max_length}, rb length {len(rb)}")
                total_count += data.numel()
                total_episodes += data["next", "done"].sum()
        # if max_length > 200:
        #    break

    t1 = time.time()

    print(
        f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
    )