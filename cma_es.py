import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import multiprocessing
from multiprocessing import Pool

import gymnasium as gym
import sys
import cma

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
    
    def get_params(self):
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
        self.n = 0
        self.mean = np.zeros(num_inputs)
        self.ssd = np.zeros(num_inputs)
    
    def partial_fit(self, X):
        self.n += 1
        delta = X - self.mean
        self.mean += delta / self.n
        delta2 = X - self.mean
        self.ssd += delta * delta2
    
    def transform(self, X):
        m = self.mean
        v = (self.ssd / self.n).clip(min=1e-2)
        s = np.sqrt(v)
        return (X - m) / s
    
scaler = OnlineStandardScaler(D)



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
    reward_per_iteration = []
    best_avg_reward = -np.inf
    
    # create optmizer
    params = initial_params
    options = {
        #'popsize': population_size,
        'AdaptSigma': True,
        'maxiter': num_iters,
        'verb_disp': 1,
        'CMA_diagonal': False, # full covariance matrix
        #'CMA_mirrors': True, # use f(params+noise) and f(params-noise) to estimate gradient
    }
    es = cma.CMAEvolutionStrategy(params, sigma, options)
    
    while not es.stop():
        t0 = datetime.now()
        offspring = es.ask()
        
        ### slow way
        # R = np.zeros(population_size)
        # for i in range(population_size):
        # R[i] = f(params + sigma * eps[i])
        
        ### fast way
        R = pool.map(f, offspring)
        R = np.array(R)
        
        es.tell(offspring, -R) # CMA-ES minimizes
        es.logger.add()
        reward_per_iteration.append(R.mean())

        print("Iter: ", es.countiter, "Avg Reward:", R.mean(), "Max Reward:", R.max(), "Duration:", datetime.now() - t0)
        
        # save best params if better than best avg reward so far
        if R.mean() > best_avg_reward:
            best_avg_reward = R.mean()
            best_params = np.mean(offspring, axis=0)
            assert(len(best_params) == num_params)
        
    return best_params, reward_per_iteration
    
def reward_function(params, record=False):
    # run one episode of env w/ params
    model = ANN(D, M, K)
    model.set_params(params)
    
    if record:
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda eps: True)
    else:
        env = gym.make(ENV_NAME)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # play one episode and return total reward
    episode_reward = 0
    episode_length = 0
    done = False
    state, _ = env.reset()
    while not done:
        scaler.partial_fit(state)
        state = scaler.transform(state)
        
        # get action
        action = model.sample_action(state)
        
        # perform action
        state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # update total reward and length
        episode_reward += reward
        episode_length += 1

    # close env
    env.close()
    
    #assert(info['episode']['r'] == episode_reward)
    return episode_reward
    

if __name__ == '__main__':
    # create model
    model = ANN(D, M, K)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # play with a saved model
        j = np.load('es_mujoco_results_cma.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        
        # in case initial shapes are not correct
        # D, M = j['W1'].shape
        # K = len(j['b2'])
        # model.D, model.M, model.K = D, M, K
    else:
        # pool for parallel evaluation
        pool = Pool(4)
        
        # train and save model
        model.init()
        params = model.get_params()
        best_params, rewards = evolution_strategy(
            f=reward_function,
            population_size=100,
            sigma=0.1,
            lr=0.01,
            initial_params=params,
            num_iters=300,
            pool=pool
        )
        
        # plot the rewards per iteration
        plt.plot(rewards)
        plt.show()
        
        #save params
        model.set_params(best_params)
        np.savez(
            "es_mujoco_results_cma.npz",
            train=rewards,
            **model.get_params_dict()
        )
    
    # play with saved model / test episode
    print("Test:", reward_function(best_params, record=True))