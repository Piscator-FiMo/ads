from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import pandas as pd
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv, ParallelEnv,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import OneHotCategorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from env.ad_optimization import AdOptimizationEnv
from sklearn.preprocessing import StandardScaler


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


    return df


def load_data() -> pd.DataFrame:
    return pd.read_csv("data/preprocessed.csv")

VERBOSE = False


def create_env():
    dataset = add_generated_synthetic_data(preprocessed_data=preprocessed_data, seed=42)
    dataset = dataset[dataset["keyword"] == "iPhone"]
    scaler = StandardScaler()
    dataset['ad_roas'] = scaler.fit_transform(dataset[['ad_roas']])

    print(dataset.describe())
    base_env = AdOptimizationEnv(dataset, initial_budget=1_000_000)
    transformed_env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    transformed_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    return transformed_env


if __name__ == "__main__":
    # Load synthetic dataset
    preprocessed_data = load_data()


    feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", "organic_ctr",
                       "paid_clicks", "paid_ctr", "ad_spend", "ad_conversions", "ad_roas", "conversion_rate",
                       "cost_per_click"]
    is_fork =False #todo remove
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0

    frames_per_batch = 10000
    # For a complete training, bring the number of frames up to 1M
    total_frames = 1_000_000
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    create_env()

    env = ParallelEnv(num_workers=10, create_env_fn=create_env)
    if VERBOSE:
        print("normalization constant shape:", env.transform[0].loc.shape)
        print("observation_spec:", env.observation_spec)
        print("reward_spec:", env.reward_spec)
        print("input_spec:", env.input_spec)
        print("action_spec (as defined by input_spec):", env.action_spec)

    check_env_specs(env)

    in_features = 13
    actor_net = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_cells, device=device),
        nn.Tanh(),
        nn.Linear(in_features=num_cells, out_features=num_cells, device=device),
        nn.Tanh(),
        nn.Linear(in_features=num_cells, out_features=env.action_spec.shape[-1], device=device)
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )


    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True
    )

    value_net = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=num_cells, device=device),
        nn.Tanh(),
        nn.Linear(in_features=num_cells, out_features=num_cells, device=device),
        nn.Tanh(),
        nn.Linear(in_features=num_cells, out_features=1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()