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

from Enhanced_Agent import QLearningClassifier, DQNAgent

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


def plot_hist(series, xlabel, title, bins=10):
    plt.figure()
    # if categorical series
    if series.dtype == object or isinstance(series.iloc[0], str):
        series.value_counts().reindex(series.unique(), fill_value=0).plot(kind='bar')
    else:
        plt.hist(series, bins=bins)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
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
def load_and_split(path="Result_stats_Clear_OffLine_20250528_013425.csv", test_size=0.2, random_state=42):
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
    plt.savefig("plots/confusion_matrix_qclassifier.png")
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
    plt.savefig("plots/classification_metrics_qclassifier.png")
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
    plt.savefig("plots/training_class_distribution.png")
    plt.close()


def train_dqn(
    df,
    agent: DQNAgent,
    classifier: QLearningClassifier,
    batch_size=32,
    epochs=5,
    eps_start=1.0,
    eps_end=0.30   ):
    import numpy as np
    import tensorflow as tf

    # 1) Build states (same as before)
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

    # 2) Rebuild model if needed
    if agent.model.input_shape[-1] != dim:
        agent.model = agent.build_model(dim)
        agent.optimizer = tf.keras.optimizers.Adam(1e-3)
        agent.loss_function = tf.keras.losses.MeanSquaredError()

    loss_history = []
    reward_history = []
    log = {k: [] for k in ['loss','delay','throughput','jitter','util','epsilon','explore','action','q_vals']}

    # total number of batches over all epochs
    num_batches = int(np.ceil(n / batch_size)) * epochs

    # 3) Custom reward as before
  
    def compute_reward(delay, packet_loss, jitter, utilization, action_name):
        # base tiers (shift +3 so worst is +1)
        if delay < 70 and packet_loss < 0.01 and jitter < 20 and utilization < 0.8:
            r = 6  # was 3+3
        elif delay < 120 and packet_loss < 0.02 and jitter < 30 and utilization < 0.9:
            r = 5  # was 2+3
        elif delay < 150 and packet_loss < 0.05 and jitter < 50 and utilization < 0.95:
            r = 4  # was 1+3
        else:
            r = 1  # was -2+3
        # congestion bonus/penalty (shifted by +2)
        congested = (delay > 120 or packet_loss > 0.02 or jitter > 30 or utilization > 0.9)
        if congested:
            if 'reroute' in action_name:       r += 4  # was +2
            elif 'decrease_bw' in action_name: r += 3  # was +1
            elif action_name=='hold_bw_keep_path': r += 0  # was -3
        else:
            if action_name=='hold_bw_keep_path':      r += 5  # was +2
            elif 'reroute' in action_name or 'decrease_bw' in action_name:
                r += 1  # was -2
        return r

    # 4) Training loop
    step = 0
    for ep in range(epochs):
        # shuffle once per epoch
        idx = np.random.permutation(n)
        Xs = X[idx]

        for b in range(0, n, batch_size):
            batch_x = Xs[b : b + batch_size]
            eps = eps_start - (eps_start - eps_end) * (step / max(1, num_batches-1))
            eps = max(eps, eps_end)          # enforce floor
            log['epsilon'].append(eps)

            actions_idx = []
            batch_rewards = []
            explores = []

            for s in batch_x:
                # choose action with current eps
                action = agent.choose_action(s, epsilon=eps, log_dict=log)
                aidx   = agent.get_action_index(action)
                actions_idx.append(aidx)

                # compute whether this was an exploration
                explores.append(int(np.random.rand() < eps))

                # compute reward
                loss_val, delay_val, _, jitter_val, util_val = s[:5]
                packet_loss = loss_val
                r = compute_reward(delay_val, packet_loss, jitter_val, util_val, action)
                batch_rewards.append(r)

            # log percent exploration this batch
            log['explore'].append(np.mean(explores))
            reward_history.append(np.mean(batch_rewards))

            # train on the batch
            loss_val = agent.train(
                np.array(batch_x, dtype=np.float32),
                np.array(actions_idx, dtype=np.int32),
                np.array(batch_rewards, dtype=np.float32)
            )
            loss_history.append(loss_val)
            step += 1

    return loss_history, reward_history, log


# --- 6) Main pipeline ---
def main():
    train_df, test_df = load_and_split()

    # Q-Learning classifier
    qagent = QLearningClassifier(fp_penalty=1.0, lr=0.1, gamma=0.9)
    q_rewards = train_qclassifier(train_df, qagent, n_episodes=200, eps_start=1.0, eps_end=0.05, samples_per_class=50)
    with open("qtable.json", "w") as f:
        json.dump(serialize_q_table(qagent.q_table), f, indent=2)
    print("qtable.json written")
    eval_qclassifier(test_df, qagent)
    plot_class_distribution(train_df)

    

    
    plot_curve(q_rewards, 'Episode', 'Reward', 'Cumulative Reward Classification-Agent', save_path="plots/training_reward_Classification.png")

    
    dqn = DQNAgent(input_dim=5 + len(qagent.classes))
    dqn_losses, rewards,  dqn_log = train_dqn(
            train_df, dqn, qagent,
            batch_size=32,
            epochs=20,
            eps_start=1.0,
            eps_end=0.30 
        )
    
    weights_path = "./dqn_model.weights.h5"
    #dqn.model.save_weights(weights_path)
    #print(f"✔ Saved DQN weights to {weights_path}")
    weights = dqn.model.get_weights()
    np.savez(
        "./dqn_model_weights.npz",
        *weights
    )
    print("✔ Saved DQN weights to dqn_model_weights.npz")
    exp = pd.Series(dqn_log['explore'])
    smooth = exp.rolling(window=500, min_periods=1, center=True).mean()
    cum_rewards = np.cumsum(rewards)
    plot_curve(cum_rewards, "Batch", "Mean Reward", "DQN Reward Curve", save_path="plots/dqn_reward_curve.png")
    plot_curve(dqn_losses, "Batch", "MSE Loss", "DQN Training Loss", save_path="plots/dqn_training_loss.png")
    plot_curve(dqn_log["epsilon"], "Step", "ε", "Epsilon Decay", save_path="plots/epsilon_decay.png")
    plot_curve(smooth, "Step", "% Explore (rolling)", "Exploration Rate", save_path="plots/exploration_rate.png")
    plot_hist(pd.Series(dqn_log["action"]), "Action", "DQN Action Frequencies")


if __name__ == '__main__':
    main()



