#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

# disable GPU (avoid CUDA init errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Enhanced_Agent import QLearningClassifier, DQNAgent, DDQNAgent



def balance_dataframe_by_label(df):
    """
    Given a DataFrame with “avg_loss_pct” and “delay” columns,
    produce a new DataFrame where each Q-class (Efficient / Loss-Degraded / etc.)
    appears equally often (resampled with replacement).
    """
    # 1) Compute a Series of labels for every row
    labels = df.apply(lambda r: derive_label(r["avg_loss_pct"]/100, r["delay"]), axis=1)
    
    # 2) Count how many rows per label, then find the minimum count
    counts = labels.value_counts()
    min_count = int(counts.min())
    
    # 3) For each label, resample that many rows (with replacement)
    balanced_parts = []
    for lbl in counts.index:
        # select only the rows whose derived-label == lbl
        subset = df[labels == lbl]
        # resample exactly min_count rows (with replacement)
        sampled = resample(subset, replace=True, n_samples=min_count, random_state=42)
        balanced_parts.append(sampled)
    
    # 4) Concatenate and shuffle
    balanced_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced_df


# --- Plotting utilities ---
def plot_curve(data, xlabel, ylabel, title, save_path=None):
    plt.figure()
    plt.plot(data, marker='o' if len(data) < 200 else None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_hist(series, xlabel, title, save_path, bins=10):
    """
    Plots a bar chart of action frequencies, using custom labels for each action.
    """
    action_labels = {
        'hold_bw_keep_path':      'maintain_bw_maintain_path',
        'decrease_bw_keep_path':  'decrease_bw_maintain_path',
        'decrease_bw_reroute':    'decrease_bw_reroute_optimal',
        'increase_bw_reroute':    'increase_bw_reroute_imm',
        'hold_bw_reroute':        'maintain_bw_reroute_optimal'
    }

    # 1) Map raw series values → friendly labels
    #    If an action isn’t in the dict, leave it unchanged.
    mapped_series = series.map(lambda x: action_labels.get(x, x))

    # 2) Count occurrences of each mapped label
    #    We want bars in the order of action_labels.values(), so we reindex accordingly.
    ordered_labels = list(action_labels.values())
    counts = mapped_series.value_counts().reindex(ordered_labels, fill_value=0)

    # 3) Plot as a bar chart
    plt.figure()
    counts.plot(kind='bar')
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 4) Save and display
    plt.savefig("plots/DQN_Actions_distribution_DDQN.png")
    plt.close()
    plt.show()





def serialize_q_table(q_table):
    """
    Turn the flat QLearningClassifier.q_table dict
    into a nested JSON‐friendly map {state_str: {action_str: value}}
    """
    nested = {}
    for (state, action), value in q_table.items():
        s = str(state)
        a = str(action)
        nested.setdefault(s, {})[a] = value
    return nested

# --- 1) Data loading & splitting ---
def load_and_split(path="Result_stats_Clear_OffLine_20250528_131238.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path).fillna(0)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

# --- 2) Derive labels for Q-Learning ---
def derive_label(loss, delay):
    if loss < 0.5 and delay < 70:
        return "Efficient"
    elif loss >= 0.5 and delay < 70:
        return "Loss-Degraded"
    elif loss < 0.5 and delay >= 70:
        return "Delay-Degraded"
    else:
        return "Congested"

# --- 3) Train Q-learning classifier ---
def train_qclassifier(
        df,
        classifier,
        n_episodes=400,
        eps_start=1.0,
        eps_end=0.05,
        samples_per_class=None
    ):
    labels = classifier.classes
    counts = df.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay']), axis=1).value_counts().reindex(labels).fillna(0)
    min_count = int(counts.min())
    balanced = []
    for lbl in labels:
        subset = df[df.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay'])==lbl, axis=1)]
        balanced.append(resample(subset, replace=True, n_samples=min_count, random_state=42))
    df_bal = pd.concat(balanced).reset_index(drop=True)

    episode_rewards = []
    for ep in range(n_episodes):
        classifier.update_epsilon(ep, n_episodes, eps_start, eps_end)
        ep_df = df_bal.copy()
        if samples_per_class:
            subdfs = []
            for lbl in labels:
                grp = df_bal[df_bal.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay'])==lbl, axis=1)]
                subdfs.append(grp.sample(n=samples_per_class, replace=True))
            ep_df = pd.concat(subdfs)
        total_r = 0
        for _, row in ep_df.iterrows():
            loss = row['avg_loss_pct']/100.0
            delay = row['delay']
            actual = derive_label(loss, delay)
            pred = classifier.classify_state(loss, delay)
            r = classifier.get_reward(pred, actual)
            classifier.train(loss, delay, pred, r)
            total_r += r
        episode_rewards.append(total_r)
    return episode_rewards

# --- 4) Evaluate Q-learning classifier ---
def eval_qclassifier(df, classifier, balance_eval=True):
    if balance_eval:
        labels = classifier.classes
        counts = df.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay']), axis=1).value_counts().reindex(labels).fillna(0)
        min_count = int(counts.min())
        balanced = []
        for lbl in labels:
            subset = df[df.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay'])==lbl, axis=1)]
            balanced.append(resample(subset, replace=True, n_samples=min_count, random_state=42))
        df = pd.concat(balanced)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        loss = row['avg_loss_pct']/100.0
        delay = row['delay']
        y_true.append(derive_label(loss, delay))
        y_pred.append(classifier.classify_state(loss, delay))
    # === Text Classification Report ===
    print("=== Q-Learning Classification Report ===")
    report = classification_report(y_true, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{label:15s} P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1-score']:.2f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=classifier.classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix_qclassifier_DDQN.png")
    plt.close()

    # === Bar Plot: Precision, Recall, F1-Score ===
    labels = list(classifier.classes)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels)

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, prec, width, label='Precision')
    plt.bar(x,         rec, width, label='Recall')
    plt.bar(x + width, f1,  width, label='F1-Score')
    plt.xticks(x, labels)
    plt.title("Classification Metrics - Q-Learning Agent")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/classification_metrics_qclassifier_DDQN.png")
    plt.close()




def plot_class_distribution(df):
    labels = df.apply(lambda r: derive_label(r['avg_loss_pct']/100, r['delay']), axis=1)
    label_counts = labels.value_counts()
    label_counts.plot(kind='bar', color='skyblue')
    plt.title("Training Data Class Distribution")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/training_class_distributionu_DDQN.png")
    plt.close()

def train_ddqn(
    df,
    agent: DDQNAgent,
    classifier: QLearningClassifier,
    batch_size=32,
    epochs=5,
    eps_start=1.0,
    eps_end=0.05
):
    import numpy as np

    # 1) Build states array exactly as before
    states = []
    classes = classifier.classes
    for _, row in df.iterrows():
        loss = row['avg_loss_pct'] / 100.0
        delay = row['delay']
        thr = row.get('total_bytes', 0)
        jit = abs(row.get('rx_bytes', 0) - row.get('tx_bytes', 0))
        util = row.get('free_bw', 0) / 100.0
        cls = classifier.classify_state(loss, delay)
        onehot = [1 if cls == c else 0 for c in classes]
        states.append([loss, delay, thr, jit, util] + onehot)

    X = np.array(states, dtype=np.float32)
    n, dim = X.shape

    # 2) If the agent’s networks weren’t built with this dimension, rebuild them
    if agent.online_model.input_shape[-1] != dim:
        agent.online_model = agent.build_model(dim)
        agent.target_model = clone_model(agent.online_model)
        agent.target_model.set_weights(agent.online_model.get_weights())
        agent.optimizer = tf.keras.optimizers.Adam(agent.lr)
        agent.loss_function = tf.keras.losses.MeanSquaredError()

    loss_history = []
    reward_history = []
    log = {k: [] for k in ['loss','delay','throughput','jitter','util','epsilon','explore','action','q_vals']}

    # Total number of mini‐batches across all epochs
    num_batches = int(np.ceil(n / batch_size)) * epochs
    step = 0

    # 3) Define the same custom reward function you used for DQN
    def compute_reward(delay, packet_loss, jitter, utilization, action_name):
        # base tiers (shift +3 so worst is +1)
        if delay < 70 and packet_loss < 0.01 and jitter < 20 and utilization < 0.8:
            r = 6
        elif delay < 120 and packet_loss < 0.02 and jitter < 30 and utilization < 0.9:
            r = 5
        elif delay < 150 and packet_loss < 0.05 and jitter < 50 and utilization < 0.95:
            r = 4
        else:
            r = 1
        congested = (delay > 120 or packet_loss > 0.02 or jitter > 30 or utilization > 0.9)
        if congested:
            if 'reroute' in action_name:
                r += 4
            elif 'decrease_bw' in action_name:
                r += 3
            elif action_name == 'hold_bw_keep_path':
                r += 0
        else:
            if action_name == 'hold_bw_keep_path':
                r += 5
            elif 'reroute' in action_name or 'decrease_bw' in action_name:
                r += 1
        return r

    # 4) Training loop
    for ep in range(epochs):
        idx = np.random.permutation(n)
        Xs = X[idx]

        for b in range(0, n, batch_size):
            batch_x = Xs[b : b + batch_size]

            # Linear ε schedule
            eps = eps_start - (eps_start - eps_end) * (step / max(1, num_batches - 1))
            eps = max(eps, eps_end)
            log['epsilon'].append(eps)

            actions_idx = []
            batch_rewards = []
            explores = []

            for s in batch_x:
                action = agent.choose_action(s, epsilon=eps, log_dict=log)
                aidx = agent.get_action_index(action)
                actions_idx.append(aidx)

                explores.append(int(np.random.rand() < eps))

                # Extract features, compute custom reward
                loss_val, delay_val, _, jitter_val, util_val = s[:5]
                packet_loss = loss_val
                r = compute_reward(delay_val, packet_loss, jitter_val, util_val, action)
                batch_rewards.append(r)

            log['explore'].append(np.mean(explores))
            reward_history.append(np.mean(batch_rewards))

            # Perform one DDQN gradient step
            loss_val = agent.train(
                np.array(batch_x, dtype=np.float32),
                np.array(actions_idx, dtype=np.int32),
                np.array(batch_rewards, dtype=np.float32)
            )
            loss_history.append(loss_val)

            step += 1

    return loss_history, reward_history, log

def main():
    train_df_raw, test_df = load_and_split()

    # Balance training set
    train_df = balance_dataframe_by_label(train_df_raw)

    # 1) Q‐Learning Classifier
    qagent = QLearningClassifier(fp_penalty=1.0, lr=0.1, gamma=0.9)
    q_rewards = train_qclassifier(
        train_df, qagent,
        n_episodes=200,
        eps_start=1.0,
        eps_end=0.05,
        samples_per_class=50
    )
    with open("qtable.json", "w") as f:
        json.dump(serialize_q_table(qagent.q_table), f, indent=2)
    print("✔ Saved Q‐table to qtable.json")
    eval_qclassifier(test_df, qagent)
    plot_class_distribution(train_df)
    plot_curve(
        q_rewards,
        "Episode",
        "Reward",
        "Cumulative Reward (Classification Agent)",
        save_path="plots/training_reward_Classification.png"
    )

    # 2) Double‐DQN Decision Agent
    input_dim_ddqn = 5 + len(qagent.classes)
    ddqn = DDQNAgent(
        eps_start=1.0,
        eps_end=0.05,
        input_dim=input_dim_ddqn,
        hidden_dims=(128, 128),
        lr=3e-4,
        gamma=0.99,
        target_update_freq=500
    )

    ddqn_losses, ddqn_rewards, ddqn_log = train_ddqn(
        df=train_df,
        agent=ddqn,
        classifier=qagent,
        batch_size=32,
        epochs=20,
        eps_start=1.0,
        eps_end=0.05
    )

    weights_path = "/home/kboussaoud/mawi_data/New_test/ddqn_model.weights.h5"
    #dqn.model.save_weights(weights_path)
    #print(f"✔ Saved DDQN weights to {weights_path}")
    weights = ddqn.online_model.get_weights()
    np.savez(
        "/home/kboussaoud/mawi_data/New_test/ddqn_model_weights.npz",
        *weights
    )

    # Plot DDQN cumulative reward
    cum_ddqn_rewards = np.cumsum(ddqn_rewards)
    plot_curve(
        cum_ddqn_rewards,
        "Batch",
        "Cumulative Reward",
        "DDQN Cumulative Reward (Decision Agent)",
        save_path="plots/training_reward_Decision.png"
    )

    # ε‐Decay & Exploration rate
    smooth_explore = pd.Series(ddqn_log["explore"]).rolling(window=500, min_periods=1, center=True).mean()
    plt.figure(figsize=(8, 5))
    plt.plot(ddqn_log["epsilon"], label="ε (epsilon)", linewidth=1)
    plt.plot(smooth_explore, label="% Explore (rolling)", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("ε‐Decay and Exploration Rate (DDQN)")
    plt.legend(loc="upper right")
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/epsilon_and_explore_together_ddqn.png")
    plt.close()

    # Action‐frequency histogram
    plot_hist(
        pd.Series(ddqn_log["action"]),
        "Action",
        "DDQN Action Frequencies",
        save_path="plots/ddqn_action_frequencies.png"
    )


if __name__ == '__main__':
    main()
