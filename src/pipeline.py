import pandas as pd
import numpy as np
import math

from src.config import config

theta0 = [None]
theta = [None]


def initialize_layer_biases(num_units):
  return np.random.uniform(low=-1,high=1,size=(1,num_units))


def initialize_layer_weights(num_units_l_1, num_units_l):
  return np.random.uniform(low=-1,high=1,size=(num_units_l_1,num_units_l))


def initialize_parameters():
  global theta0, theta
  theta0 = [None]  
  theta = [None]

  for l in range(1, config.NUM_LAYERS):
        theta0_l = initialize_layer_biases(config.P[l])
        theta_l = initialize_layer_weights(config.P[l-1], config.P[l])

        theta0.append(theta0_l / math.sqrt(config.P[l-1]))
        theta.append(theta_l / math.sqrt(config.P[l-1]))


def mini_batch_training(X, Y, mini_batch_size=2):
    initialize_parameters()
    
    num_batches = math.ceil(len(X) / mini_batch_size)
    
    for epoch in range(config.NUM_INPUTS):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * mini_batch_size
            end_idx = min((batch_idx + 1) * mini_batch_size, len(X))
            
            X_batch = X[start_idx:end_idx]
            Y_batch = Y[start_idx:end_idx]
         