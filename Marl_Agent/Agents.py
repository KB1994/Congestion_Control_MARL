import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model as keras_load_model
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
    def __init__(self, eps_start=1.0, eps_end = 0.30, eps_decay=0.995, input_dim=5):
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





class DoubleDQNAgent:
    def __init__(self, state_size=5, action_size=5, gamma=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.actions = [
            'hold_bw_keep_path',
            'decrease_bw_keep_path',
            'decrease_bw_reroute',
            'increase_bw_reroute',
            'hold_bw_reroute'
        ]

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(
            self, loss, delay, throughput, jitter, utilization,
            epsilon=0.1, log_dict=None
        ):
        explore = np.random.rand() < epsilon

        if explore:
            action = random.choice(self.actions)
            q_vals = None
        else:
            state = np.array([loss, delay, throughput, jitter, utilization]).reshape(1, -1)
            q_vals = self.model(state, training=False).numpy()[0]
            action = self.actions[int(np.argmax(q_vals))]

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


    def train(self, states, actions_idx, rewards, next_states, dones):
        actions_one_hot = tf.one_hot(actions_idx, self.action_size)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_action = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            next_q_values = self.model(next_states)
            next_actions = tf.argmax(next_q_values, axis=1)

            target_q_values = self.target_model(next_states)
            selected_q_values = tf.gather(target_q_values, next_actions, axis=1, batch_dims=1)

            target = rewards + self.gamma * selected_q_values * (1 - dones)
            loss = self.loss_function(target, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action_index(self, action_name):
        return self.actions.index(action_name)

    def split_action(self, action_name):
        return tuple(action_name.split('_')[0::2])

    def save_model(self, path='saved_models/dqn.h5'):
        self.model.save(path)

    def load_model(self, path='saved_models/dqn.h5'):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()




class DoubleDQNAgent_1:
    def __init__(self, state_size=5, action_space=None):
        self.state_size = state_size
        self.actions = action_space or [
            'hold_bw_keep_path',
            'decrease_bw_keep_path',
            'decrease_bw_reroute',
            'increase_bw_reroute',
            'hold_bw_reroute'
        ]
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.actions))
        ])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        return self.actions[np.argmax(q_values)]

    def get_action_index(self, action):
        return self.actions.index(action)

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones)

        next_q = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        max_actions = np.argmax(next_q, axis=1)
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target[np.arange(len(batch)), max_actions]

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_action = tf.reduce_sum(q_values * tf.one_hot(actions, len(self.actions)), axis=1)
            loss = self.loss_fn(target_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()
