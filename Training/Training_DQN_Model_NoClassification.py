#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from marl_agent.Enhanced_Agent import DQNAgent

# ——————————————————————————————————————————
# 0) Load and derive features for X (N×5)
# ——————————————————————————————————————————

csv_path = "Result_stats_Clear_OffLine_20250528_131238.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Expected CSV at {csv_path}")

df = pd.read_csv(csv_path)


# 1) Convert avg_loss_pct (%) → loss fraction
df["loss_frac"] = df["avg_loss_pct"] / 100.0

# 2) Delay is already in ms: df["delay"]

# 3) Throughput = speed (Mbps), keep as is (or convert to float)
df["throughput"] = df["speed"].astype(float)

# 4) Jitter: simple one‐step absolute delta of speed
#    (assumes rows are ordered in time; if not, sort by time first)
df["jitter"] = df["throughput"].diff().abs().fillna(0.0)

# 5) Utilization: we can treat free_bw (Mbps) ÷ capacity (assuming 100 Mbps link)
#    so util_fraction = free_bw / 100.0
df["util"] = df["free_bw"] / 100.0

# Now assemble X as a NumPy array of shape (N, 5), columns: [loss, delay, throughput, jitter, util]
features = ["loss_frac", "delay", "throughput", "jitter", "util"]
X = df[features].to_numpy(dtype=np.float32)

# You may want to shuffle or split X here; for now, we’ll use the entire set.

# ——————————————————————————————————————————
# 1) Instantiate DQN agent
# ——————————————————————————————————————————

input_dim = 5
dqn = DQNAgent(input_dim=input_dim)
dqn.model = dqn.build_model(input_dim)
dqn.optimizer     = tf.keras.optimizers.Adam(1e-3)
dqn.loss_function = tf.keras.losses.MeanSquaredError()

# ——————————————————————————————————————————
# 2) Hyperparameters
# ——————————————————————————————————————————
batch_size = 32
epochs     = 20
eps_start  = 1.0
eps_end    = 0.05

n  = X.shape[0]
num_batches = int(np.ceil(n / batch_size)) * epochs

# ——————————————————————————————————————————
# 3) Initialize logging
# ——————————————————————————————————————————
log = {
    "loss":       [],
    "delay":      [],
    "throughput": [],
    "jitter":     [],
    "util":       [],
    "epsilon":    [],
    "explore":    [],
    "action":     [],
    "q_vals":     []
}
loss_history   = []
reward_history = []

step = 0
for ep in range(epochs):
    # Shuffle X at the start of each epoch
    idx = np.random.permutation(n)
    Xs  = X[idx]

    for b in range(0, n, batch_size):
        batch_x = Xs[b : b + batch_size]

        # ε‐decay
        eps = eps_start - (eps_start - eps_end) * (step / max(1, num_batches - 1))
        eps = max(eps, eps_end)
        log["epsilon"].append(eps)

        actions_idx   = []
        batch_rewards = []
        explores      = []

        for s in batch_x:
            # s = [loss_frac, delay, throughput, jitter, util]
            action = dqn.choose_action(s, epsilon=eps, log_dict=log)
            aidx   = dqn.get_action_index(action)
            actions_idx.append(aidx)

            explores.append(int(np.random.rand() < eps))

            loss_val, delay_val, thr_val, jitter_val, util_val = s

            # Example reward function (you can replace this with your own logic):
            def compute_reward(delay, loss, jitter, util, action):
                # e.g. negative weighted sum of delay & loss, positive for util
                return - (delay * 0.1 + loss * 10 + jitter * 0.05) + util * 5

            r = compute_reward(delay_val, loss_val, jitter_val, util_val, action)
            batch_rewards.append(r)

        log["explore"].append(np.mean(explores))
        reward_history.append(np.mean(batch_rewards))

        # Train on this batch
        loss_val = dqn.train(
            np.array(batch_x, dtype=np.float32),
            np.array(actions_idx, dtype=np.int32),
            np.array(batch_rewards, dtype=np.float32)
        )
        loss_history.append(loss_val)
        step += 1

# ——————————————————————————————————————————
# 4) Save DQN weights
# ——————————————————————————————————————————
weights = dqn.model.get_weights()
np.savez("dqn_only_weights.npz", *weights)
print("✔ Saved DQN‐only weights to dqn_only_weights.npz")

# ——————————————————————————————————————————
# 5) Plot training metrics
# ——————————————————————————————————————————

# Cumulative Reward
cum_rewards = np.cumsum(reward_history)
plt.figure()
plt.plot(cum_rewards, marker="o" if len(cum_rewards) < 200 else None)
plt.xlabel("Batch")
plt.ylabel("Cumulative Reward")
plt.title("DQN Cumulative Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_reward_dqn_no_class.png")
plt.close()

# Training Loss
plt.figure()
plt.plot(loss_history, marker="o" if len(loss_history) < 200 else None)
plt.xlabel("Batch")
plt.ylabel("MSE Loss")
plt.title("DQN Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_dqn_no_class.png")
plt.close()

# Epsilon and Exploration Rate
plt.figure()
plt.plot(log["epsilon"], label="ε (epsilon)")
plt.plot(pd.Series(log["explore"])
         .rolling(window=500, min_periods=1, center=True)
         .mean(),
         label="% Explore (rolling)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("ε‐Decay and Exploration Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("epsilon_and_explore_dqn_no_class.png")
plt.close()

# Action Distribution
actions_series = pd.Series(log["action"], name="action")
action_counts = actions_series.value_counts().sort_index()

plt.figure(figsize=(6,4))
action_counts.plot(kind="bar")
plt.xlabel("Action Index")
plt.ylabel("Frequency")
plt.title("DQN Action Distribution")
plt.tight_layout()
plt.savefig("action_distribution_dqn_no_class.png")
plt.close()

print("✔ Saved action distribution to 'action_distribution_dqn_no_class.png'")
