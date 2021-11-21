import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces

import util


class AutoRebalanceEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_price: pd.DataFrame, targets, means, covs, re_target, alpha=1, len_period: int = 500,
                 mode='train', start_tick=0):
        super(AutoRebalanceEnv, self).__init__()

        # constant
        self.MODE = mode
        self.SELL_COST = .004415
        self.BUY_COST = .001425
        self.TIME_TO_RESAMPLE = 10
        self.LEN_OF_PERIOD = len_period
        self.ALPHA = alpha
        self.stock_price = stock_price
        self.targets = targets
        self.means = means
        self.covs = covs
        self.mu = None
        self.cov = None
        self.target_ratio = None
        self.cum_target = None
        self.re_target = re_target
        self.daily_growth = util.daily_growth(self.stock_price)

        # step variable
        self.done = False
        if self.MODE == 'train':
            self.start_tick = np.random.choice(range(len(self.stock_price) - self.LEN_OF_PERIOD))
        else:
            self.start_tick = 0
        self.current_tick = self.start_tick
        self.last_tick = self.current_tick + self.LEN_OF_PERIOD - 2
        self.target_ratio = self.targets[self.current_tick]
        self.cum_target = np.cumsum(self.target_ratio)[:-1]

        self.current_weight = self.target_ratio
        self.current_return = 1.
        self.growth_rate = 1.
        self.n_train = 0
        self.n_rebalance = 0
        self.n_step = 0
        self.total_cost = 0
        self.total_error = 0
        self.total_utility_error = 0

        self.history = None

        # action and observation space
        self.n_actions = self.stock_price.shape[1]
        self.action_space = spaces.Box(low=self.cum_target - .05, high=self.cum_target + .05,
                                       shape=(self.n_actions - 1,), dtype=np.float64)
        # self.action_space = spaces.Box(low=0.01, high=1, shape=(self.n_actions, ), dtype=np.float64)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_actions * 3+1, 1), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_actions * 2, 1), dtype=np.float64)

    def step(self, action: np.ndarray):
        self.n_step += 1
        self.update_target()
        # extract action
        next_state = np.linspace(0, 1, self.n_actions + 1)
        next_state[1:-1] = action
        action = np.diff(next_state) - self.current_weight
        # next_state = action/np.sum(action)
        # action = next_state - self.current_weight

        # determine whether to move to new stage
        if np.sum(np.abs(action)) >= .01:
            self.n_rebalance += 1
            self._apply_action(action)

        self.current_tick += 1
        self.done = self.current_tick == self.last_tick

        trading_cost = self._get_trading_cost(action)

        if not self.done:
            self.current_return = self.current_weight @ (self.daily_growth[self.current_tick - 1] + 1)
            self.growth_rate *= self.current_return
            self._update_current_weight()
        utility_error = self._get_trace_error()

        # observation = np.concatenate(
        #     [self.stock_price[self.current_tick]-self.stock_price[self.current_tick-1], self.current_weight, self.target_ratio, [utility_error]]).reshape(-1, 1)
        observation = np.array([self.current_weight, self.target_ratio]).reshape(-1, 1)

        trace_error = np.sum(np.abs(observation))

        reward = - (trading_cost) + (self.current_return-1) + utility_error
        self.total_cost += trading_cost
        self.total_error += trace_error
        self.total_utility_error += utility_error

        info = {
            'Observation': observation,
            'Next State': next_state,
            'Current Weight': self.current_weight,
            'Action': action,
            '# Rebalance': self.n_rebalance,
            'Trading Cost': trading_cost,
            'Trace Error': trace_error,
            'Daily Return': self.current_return,
            'Total Cost': self.total_cost,
            'Total Error': self.total_error,
            'Total UE': self.total_utility_error,
            'Total Return': self.growth_rate,
            'Reward': reward,
        }
        self._update_history(info)
        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.n_train += 1
        self.n_rebalance = 0
        self.n_step = 0
        if self.MODE == 'train' and (self.n_train % self.re_target) == 0:
            self.start_tick = np.random.choice(range(len(self.stock_price) - self.LEN_OF_PERIOD))
        else:
            self.start_tick = 0
        self.current_tick = self.start_tick
        self.last_tick = self.current_tick + self.LEN_OF_PERIOD - 2
        self.update_target()
        self.current_weight = self.target_ratio
        self.current_return = 1.
        self.growth_rate = 1.
        self.history = None

        self.total_cost = 0
        self.total_error = 0
        self.total_utility_error = 0
        # self.action_space.reset(state=self.target_ratio)
        observation = np.array([self.current_weight, self.target_ratio]).reshape(-1, 1)


        return observation

    def render(self, mode='human'):
        if self.history is not None:
            print(f'{self.n_train} epoch:, {self.current_tick} ticker:')
            for key in self.history:
                print(f'  {key}: {self.history[key][-1]}')
        else:
            print('First Step: ')

    def close(self):
        fig = plt.figure()
        for key in ['Action']:
            print(key)
            plt.plot(self.history[key])
        plt.show()

    def _apply_action(self, action):
        self.current_weight += action

    def _update_growth_rate(self):
        self.growth_rate = self.current_weight @ (self.daily_growth[self.current_tick - 1] + 1)

    def _update_current_weight(self):
        self.current_weight *= (self.daily_growth[self.current_tick - 1] + 1)
        self.current_weight = util.norm(self.current_weight)

    def _get_trading_cost(self, action):
        tc = 0
        for a in action:
            if a < 0:
                tc += -a * self.SELL_COST
            else:
                tc += a * self.BUY_COST
        tc *= self.growth_rate
        return tc

    def update_target(self):
        if self.n_step % self.re_target == 0:
            self.target_ratio = self.targets[self.current_tick]
            self.cum_target = np.cumsum(self.target_ratio)[:-1]
            self.action_space = spaces.Box(low=self.cum_target - .05, high=self.cum_target + .05,
                                           shape=(self.n_actions - 1,), dtype=np.float32)

    def utility(self, weights):
        self.mu = self.means[self.current_tick]
        self.cov = self.covs[self.current_tick]
        return weights.T @ self.mu - 0.5 * self.ALPHA * weights.T @ self.cov @ weights

    def _get_trace_error(self):
        return self.utility(self.target_ratio) - \
               self.utility(self.current_weight)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)

    def save_history(self, filename):
        history_df = pd.DataFrame.from_dict(self.history)
        history_df.to_csv(filename)


if __name__ == '__main__':

    df = pd.read_csv('raw data/2000_2019_daily_data.csv')[
        ['H0A0 Index', 'NKY Index', 'SENSEX Index', 'SPX Index']].to_numpy()
    env = AutoRebalanceEnv(stock_price=df, target_ratio=np.array([.13, .14, .17, .56]))
    observation = env.reset()

    for _ in range(997):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()
