import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras as tfk
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras import losses
import pandas as pd
import numpy as np

class Autoencoder(Model):
    def __init__(self, n_books=10000):
        super(Autoencoder, self).__init__()
        
        self.encoder1 = layers.Dense(80, activation='tanh')
        self.encoder2 = layers.Dense(40, activation='tanh')
        self.decoder1 = layers.Dense(80, activation='relu')
        self.decoder2 = layers.Dense(n_books, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.2)
        
    def call(self, x, training=False):
        x = self.encoder1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.encoder2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.decoder1(x)
        decoded = self.decoder2(x)
        
        return decoded