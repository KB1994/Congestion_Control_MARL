import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, backend as K, Model, Input
from tensorflow.keras.models import load_model as keras_load_model

from tensorflow.keras.models import clone_model  

import os
import copy
import json
from collections import defaultdict


class QLearningClassifier:
    def __init__(
        self,
        classes=None,
        state_thresholds=(0.5, 70),  # (loss_threshold, delay_threshold)
        lr=0.1,
        gamma=0.9,
        fp_penalty=1.0
    ):
        # Define classes (can override by passing a list)
        self.classes = classes or ["Efficient", "Loss-Degraded", "Delay-Degraded", "Congested"]
        # Initialize Q-table for the 4 possible (loss,delay) buckets
        self.q_table = {
            (0, 0): [0.0] * len(self.classes),  # low loss, low delay
            (1, 0): [0.0] * len(self.classes),  # high loss, low delay
            (0, 1): [0.0] * len(self.classes),  # low loss, high delay
            (1, 1): [0.0] * len(self.classes),  # high loss, high delay
        }
        # Hyperparameters
        self.epsilon = 1.0          # will be decayed externally during training
        self.lr = lr
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        # Thresholds for discretizing loss and delay
        self.loss_thr, self.delay_thr = state_thresholds

    def _state_to_tuple(self, loss, delay):
        # Map continuous loss/delay into four discrete states
        return (int(loss >= self.loss_thr), int(delay >= self.delay_thr))

    def classify_state(self, loss, delay):
        state = self._state_to_tuple(loss, delay)
        # ε-greedy: explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.classes)
        idx = int(np.argmax(self.q_table[state]))
        return self.classes[idx]

    def get_reward(self, predicted_class, actual_class):
        # +1 for correct, -1 for incorrect
        base = 1 if predicted_class == actual_class else -1
        # Additional penalty for false alarms on non-Efficient classes
        if predicted_class != actual_class and predicted_class != "Efficient":
            base -= self.fp_penalty
        return base

    def train(self, loss, delay, predicted_class, reward):
        state = self._state_to_tuple(loss, delay)
        idx = self.classes.index(predicted_class)
        current_q = self.q_table[state][idx]
        max_future_q = max(self.q_table[state])
        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][idx] = new_q

    def update_epsilon(self, episode, max_episodes, eps_start=1.0, eps_end=0.05):
        # Linearly decay epsilon over episodes
        decay = (eps_start - eps_end) * (episode / float(max_episodes - 1))
        self.epsilon = max(eps_end, eps_start - decay)



# === DQN Decision Agent ===

class DQNAgent:
    def __init__(self, eps_start=1.0, eps_end = 1.00, eps_decay=0.995, input_dim=5):
        # Compound action space: bandwidth + routing
        self.actions = [
            'hold_bw_keep_path',
            'decrease_bw_keep_path',
            'decrease_bw_reroute',
            'increase_bw_reroute',
            'hold_bw_reroute'
        ]
        self.epsilon     = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay

        self.model = self.build_model(input_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        


    def build_model(self, input_dim):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(input_dim,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(len(self.actions)))
        return model

    def choose_action(self, state, epsilon=0.1, log_dict=None):
        # state: full vector [loss, delay, throughput, jitter, util, one-hot...]
        loss, delay, throughput, jitter, utilization = state[:5]
        explore = np.random.rand() < epsilon

        if explore:
            action = random.choice(self.actions)
            q_vals = None
        else:
            # feed the entire state (including one-hot classes) to the network
            input_vec = np.array(state, dtype=np.float32).reshape(1, -1)
            q_vals = self.model(input_vec, training=False).numpy()[0]
            action = self.actions[int(np.argmax(q_vals))]

        # logging
        if log_dict is not None:
            log_dict["loss"].append(loss)
            log_dict["delay"].append(delay)
            log_dict["throughput"].append(throughput)
            log_dict["jitter"].append(jitter)
            log_dict["util"].append(utilization)
            log_dict["epsilon"].append(epsilon)
            log_dict["explore"].append(int(explore))
            log_dict["action"].append(action)
            log_dict["q_vals"].append(q_vals if q_vals is not None else [np.nan]*len(self.actions))

        return action


    def train(self, states, actions_idx, rewards):
        actions_one_hot = tf.one_hot(actions_idx, len(self.actions))
        rewards = tf.cast(rewards, tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            q_values = tf.reduce_sum(predictions * actions_one_hot, axis=1)
            loss = self.loss_function(rewards, q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action_index(self, action_name):
        return self.actions.index(action_name)

    def split_action(self, action_name):
        """Returns (bandwidth_action, routing_action)"""
        return tuple(action_name.split('_', maxsplit=1)[0::2])  # e.g., ('hold_bw', 'keep_path')


    def save_model(self, path="dqn_model.weights.h5"):
        self.model.save_weights(path)
        print(f"✔ Saved DQN weights to {path}")

    def load_weights_npz(self, path="dqn_model_weights.npz"):
        """
        Load weights from a numpy .npz archive into the model.
        """
        npz = np.load(path)
        # npz.files is ['arr_0','arr_1',…] in insertion order
        # turn it into a list in sorted order
        arrays = [npz[f] for f in sorted(npz.files, key=lambda x: int(x.split('_')[1]))]
        self.model.set_weights(arrays)
        print(f"✔ Loaded DQN weights from {path}")


class DDQNAgent:
    """
    Double‐DQN Agent: uses two networks (online + target) and updates via the Double‐DQN rule.
    """

    def __init__(self,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=0.995,
                 input_dim=5,
                 hidden_dims=(128, 128),
                 lr=3e-4,
                 gamma=0.99,
                 target_update_freq=500):
        """
        Args:
            eps_start (float): initial ε for exploration.
            eps_end   (float): final ε for exploration.
            eps_decay (float): (unused here) placeholder if you prefer multiplicative decay.
            input_dim (int):  dimensionality of the state vector.
            hidden_dims (tuple): size of the two hidden layers.
            lr        (float): learning rate for Adam.
            gamma     (float): discount factor.
            target_update_freq (int): number of gradient steps between target‐network copies.
        """
        # Define the action space (must match DQNAgent’s actions exactly)
        self.actions = [
            'hold_bw_keep_path',
            'decrease_bw_keep_path',
            'decrease_bw_reroute',
            'increase_bw_reroute',
            'hold_bw_reroute'
        ]
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.target_update_freq = target_update_freq

        # Build the online (policy) network
        self.online_model = self.build_model(input_dim)

        # Build the target network as a clone of online_model
        self.target_model = clone_model(self.online_model)
        self.target_model.set_weights(self.online_model.get_weights())

        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.loss_function = losses.MeanSquaredError()

        

        # Internal counter for gradient steps (used to know when to sync target network)
        self._gradient_steps = 0

    def build_model(self, input_dim):
        """
        Constructs a simple feedforward Q‐network with two hidden layers.
        """
        inputs = Input(shape=(input_dim,), name="state_input")
        x = layers.Dense(self.hidden_dims[0], activation='relu')(inputs)
        x = layers.Dense(self.hidden_dims[1], activation='relu')(x)
        q_values = layers.Dense(len(self.actions), name="q_values")(x)
        model = Model(inputs=inputs, outputs=q_values)
        return model

    def choose_action(self, state, epsilon=None, log_dict=None):
        """
        Epsilon‐greedy action selection.
        Args:
            state (1‐D np.array or list): must be length = input_dim (5 raw features + one‐hot label length).
            epsilon (float): if provided, overrides self.epsilon for this call.
            log_dict (dict or None): if not None, will append logs in the same format as DQNAgent.
        Returns:
            action_name (str)
        """
        if epsilon is None:
            epsilon = self.epsilon

        explore = np.random.rand() < epsilon
        if explore:
            action = random.choice(self.actions)
            q_vals = None
        else:
            # Reshape to (1, input_dim)
            s_tensor = tf.convert_to_tensor(np.array(state, dtype=np.float32).reshape(1, -1))
            q_vals = self.online_model(s_tensor, training=False).numpy()[0]
            best_idx = int(np.argmax(q_vals))
            action = self.actions[best_idx]

        # Logging (same keys as DQNAgent)
        if log_dict is not None:
            # Raw features
            loss_val, delay_val, thr_val, jit_val, util_val = state[:5]
            log_dict.setdefault("loss", []).append(loss_val)
            log_dict.setdefault("delay", []).append(delay_val)
            log_dict.setdefault("throughput", []).append(thr_val)
            log_dict.setdefault("jitter", []).append(jit_val)
            log_dict.setdefault("util", []).append(util_val)
            log_dict.setdefault("epsilon", []).append(epsilon)
            log_dict.setdefault("explore", []).append(int(explore))
            log_dict.setdefault("action", []).append(action)
            log_dict.setdefault("q_vals", []).append(
                q_vals if q_vals is not None else [np.nan] * len(self.actions)
            )

        return action

    def get_action_index(self, action_name):
        return self.actions.index(action_name)

    def train(self, states, actions_idx, rewards):
        """
        Performs one Double‐DQN update step over the batch:
          1) Compute Q_online(s, a) for the taken actions.
          2) Compute next‐action indices from online_model on next_states (here: states again).
          3) Compute Q_target(next_states, next_actions) via target_model.
          4) Form TD target: y = r + γ * Q_target(next_states, next_actions).
          5) Compute MSE( Q_online(s, a), y ) and backpropagate on online_model.
          6) Every target_update_freq steps, copy weights online → target.
        """
        # states: shape (batch_size, input_dim)
        # actions_idx: shape (batch_size,)
        # rewards: shape (batch_size,)
        batch_size = states.shape[0]

        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)      # (batch, input_dim)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)    # (batch,)

        # In this offline setting, we treat “next_states” = “states” again
        next_states_tensor = states_tensor

        with tf.GradientTape() as tape:
            # 1) Q_online(s, all_actions)
            q_online_all = self.online_model(states_tensor, training=True)  # (batch, A)

            # Extract Q_online(s, a_taken)
            action_one_hot = tf.one_hot(actions_idx, depth=len(self.actions), dtype=tf.float32)  # (batch, A)
            q_online_taken = tf.reduce_sum(q_online_all * action_one_hot, axis=1)  # (batch,)

            # 2) Q_online(next_states, all_actions) → used to pick next_action per sample
            q_online_next_all = self.online_model(next_states_tensor, training=False)  # (batch, A)
            next_actions = tf.argmax(q_online_next_all, axis=1, output_type=tf.int32)   # (batch,)

            # 3) Q_target(next_states, next_actions)
            q_target_next_all = self.target_model(next_states_tensor, training=False)  # (batch, A)
            next_action_one_hot = tf.one_hot(next_actions, depth=len(self.actions), dtype=tf.float32)  # (batch, A)
            q_target_next = tf.reduce_sum(q_target_next_all * next_action_one_hot, axis=1)  # (batch,)

            # 4) TD target: r + γ * Q_target(next_states, next_actions)
            td_target = rewards_tensor + self.gamma * q_target_next  # shape (batch,)

            # 5) Loss = MSE( td_target, Q_online(s, a_taken) )
            loss = self.loss_function(td_target, q_online_taken)  # scalar

        # 6) Backpropagate on online_model
        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_model.trainable_variables))

        # 7) Update target network once every target_update_freq gradient steps
        self._gradient_steps += 1
        if self._gradient_steps % self.target_update_freq == 0:
            self.target_model.set_weights(self.online_model.get_weights())

        return loss.numpy()

    def save_model(self, path='ddqn_agent.weights.h5'):
        """
        Save only the online network’s weights for lightweight deployment.
        """
        # Make sure the directory exists
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self.online_model.save_weights(path)
        print(f"✔ Saved DDQN online‐network weights to {path}")

   

    def save_model(self, path="ddqn_model.weights.h5"):
        self.model.save_weights(path)
        print(f"✔ Saved DDQN weights to {path}")

    def load_weights_npz(self, path="ddqn_model_weights.npz"):
        """
        Load weights from a numpy .npz archive into the model.
        """
        npz = np.load(path)
        # npz.files is ['arr_0','arr_1',…] in insertion order
        # turn it into a list in sorted order
        arrays = [npz[f] for f in sorted(npz.files, key=lambda x: int(x.split('_')[1]))]
        self.online_model.set_weights(arrays)
        print(f"✔ Loaded DDQN weights from {path}")
