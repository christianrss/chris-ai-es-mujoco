import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import multiprocessing
from multiprocessing import Pool

import gymnasium as gym
import sys

# enviorment
ENV_NAME = 'HalfCheetah-v4'

### neural network

# hyperparameters
env = gym.make(ENV_NAME)
D = np.prod(env.observation_space.shape)
M = 128
K = env.action_space.shape[0]
action_max = env.action_space.high[0]

def relu(x):
    return x * (x > 0)

class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.f = f
        
    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)
        
    def forward(self, X):
        Z = self.f(X @ self.W1 + self.b1)
        return np.tanh(Z @ self.W2 + self.b2) * action_max
    
    def sample_action(self, x):
        # assume input is a single state of size (D,)
        # firstmake it (N, D) to fit ML conventions
        X = np.atleast_2d(x)
        Y = self.forward(X)
        return Y[0] # first row
    
    def get_params():
        # return a flat array of parameters
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
    
    def get_params_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        
    def set_params(self, params):
        # params is a flat list
        # unflatten into individual weights
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M: D * M + M]
        self.W2 = params[D * M + M: D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]

class OnlineStandardScaler:
    def __init__(self, num_inputs):
        pass
    
scaler = OnlineStandardScaler(D)

class Adam:
    pass

def evolution_strategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters,
    pool):
    
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    
    # create optmizer
    params = initial_params
    adam = Adam(params, lr)
    
    for t in range(num_iters):
        t0 = datetime.now()
        eps = np.random.randn(population_size, num_params)
        
        ### slow way
        # R = np.zeros(population_size)
        # for i in range(population_size):
        # R[i] = f(params + sigma * eps[i])
        
        ### fast way
        R = pool.map(f, [params + sigma * eps[i] for i in range(population_size)])
        R = np.array(R)
        
        m = R.mean()
        s = R.std()
        if s == 0:
            # we can't apply the folowing equation
            print("Skipping")
            continue

        A = (R - m) / s
        reward_per_iteration[t] = m
        g = eps.T @ A / (population_size * sigma)
        params= adam.update(g)

        print("Iter: ", t, "Avg Reward:", m, "Max Reward:", R.max(), "Duration:", datetime.now() - t0)
        
    return reward_per_iteration
    
def reward_function(params):
    # run one episode of env w/ params
    pass

if __name__ == '__main__':
    pool = Pool(4)