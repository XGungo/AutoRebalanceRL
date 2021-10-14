import gym
import pandas as pd
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import util


class RebalanceBox(spaces.Box):
    def __init__(self, low, high, shape, dtype, num_commodities, state):
        super(RebalanceBox, self).__init__(low, high, shape, dtype)
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.num_commodities = num_commodities
        self.current_state = state
        self.delta = .02
        self.update_rate = .99

    def reset(self, state, delta=.1) -> None:
        self.current_state = state
        self.delta = delta

    def update(self, state: np.ndarray) -> None:
        self.current_state = state

    # def sample(self) -> np.ndarray:
    #     accumulated_state = np.cumsum(self.current_state)
    #     accumulated_state[:-1] += np.random.normal(0, self.delta, len(accumulated_state) - 1)
    #     accumulated_state = np.insert(accumulated_state, 0, 0)
    #     accumulated_state = np.diff(accumulated_state) - self.current_state.flatten()
    #     self.delta *= self.update_rate
    #     return accumulated_state.reshape(self.shape)

    def sample(self) -> np.ndarray:
        accumulated_new_state = np.linspace(0, 1, self.num_commodities + 1)
        accumulated_new_state[1:-1] = np.sort(np.random.uniform(0, 1, self.num_commodities - 1))
        new_state = np.diff(accumulated_new_state)
        return new_state - self.current_state


class AutoRebalanceEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha, stock_price: pd.DataFrame, target_ratio: np.ndarray, len_period: int = 500, mode='test'):
        super(AutoRebalanceEnv, self).__init__()

        # constant
        self.MODE = mode
        self.SELL_COST = .004415
        self.BUY_COST = .001425
        self.n_train = 0
        self.n_rebalance = 0
        self.TIME_TO_RESAMPLE = 10
        self.LEN_OF_PERIOD = len_period
        self.ALPHA = alpha
        self.stock_price = stock_price
        self.TARGET_RATIO = target_ratio
        self.CUMSUM_TARGET = np.cumsum(self.TARGET_RATIO)[:-1]
        self.daily_growth = util.daily_growth(self.stock_price)
        self.cov = np.cov(self.daily_growth.T)
        self.mu = util.expected_mean_return(self.stock_price)*self.LEN_OF_PERIOD

        # step variable
        self.done = False
        self.start_tick = np.random.choice(len(self.stock_price) - self.LEN_OF_PERIOD) if self.MODE == 'train' else 0
        self.current_tick = self.start_tick
        self.last_tick = self.current_tick + self.LEN_OF_PERIOD - 1

        self.current_weight = self.TARGET_RATIO
        self.current_return = 1.
        self.growth_rate = 1.

        self.total_cost = 0
        self.total_error = 0
        self.total_utility_error = 0

        self.history = None

        # action and observation space
        self.n_actions = stock_price.shape[1]
        # self.action_space = RebalanceBox(low=0, high=1, shape=(self.n_actions, ),
        #                                  dtype=np.float32, num_commodities=self.n_actions, state=self.TARGET_RATIO)
        self.action_space = spaces.Box(low=self.CUMSUM_TARGET-.05, high=self.CUMSUM_TARGET+.05,
                                       shape=(self.n_actions-1, ), dtype=np.float32)
        # self.action_space = spaces.Box(low=0, high=1,
        #                                shape=(self.n_actions - 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_actions, ), dtype=np.float32)

    def step(self, action: np.ndarray):
        next_state = np.linspace(0, 1, self.n_actions+1)
        next_state[1:-1] = action
        action = np.diff(next_state) - self.current_weight
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
        observation = self.TARGET_RATIO - self.current_weight
        trace_error = np.sum(np.abs(observation))
        reward = - (trace_error + trading_cost) + self.current_return

        self.total_cost += trading_cost
        self.total_error += trace_error
        self.total_utility_error += utility_error

        info = {
            'Observation': observation,
            'Next State': np.diff(next_state),
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
        # self.action_space.update(self.current_weight)
        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.n_train += 1
        self.n_rebalance = 0
        if self.n_train % self.TIME_TO_RESAMPLE == 0:
            self.start_tick = np.random.choice(len(self.stock_price) - self.LEN_OF_PERIOD) if self.MODE == 'train' else 0
        self.current_tick = self.start_tick
        self.last_tick = self.current_tick + self.LEN_OF_PERIOD - 1
        self.current_weight = self.TARGET_RATIO
        self.current_return = 1.
        self.growth_rate = 1.
        self.history = None

        self.total_cost = 0
        self.total_error = 0
        self.total_utility_error = 0
        # self.action_space.reset(state=self.TARGET_RATIO)
        observation = self.TARGET_RATIO - self.current_weight

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

    @staticmethod
    def utility(weights, mu, cov, alpha):
        return np.dot(mu, weights) - 0.5 * alpha * np.dot(weights.T, np.dot(cov, weights))

    def _get_trace_error(self):
        return AutoRebalanceEnv.utility(self.TARGET_RATIO, self.mu, self.cov, self.ALPHA) - \
               AutoRebalanceEnv.utility(self.current_weight, self.mu, self.cov, self.ALPHA)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)

    def save_history(self, filename):
        history_df = pd.DataFrame.from_dict(self.history)
        history_df.to_csv(filename)


if __name__ == '__main__':

    df = pd.read_csv('2000_2019_daily_data.csv')[['H0A0 Index', 'NKY Index', 'SENSEX Index', 'SPX Index']].to_numpy()
    env = AutoRebalanceEnv(stock_price=df, target_ratio=np.array([.13, .14, .17, .56]))
    observation = env.reset()

    for _ in range(997):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()

