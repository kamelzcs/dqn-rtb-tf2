import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import layers


class DDPG_Network:
    def __init__(self, num_states=5, num_actions=1, lower_bound=-0.1, upper_bound=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(50, activation="relu")(inputs)
        out = layers.Dense(50, activation="relu")(out)
        out = layers.Dense(50, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(50, activation="relu")(concat)
        out = layers.Dense(50, activation="relu")(out)
        out = layers.Dense(50, activation="relu")(out)
        outputs = layers.Dense(1, activation=tf.nn.relu)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


