import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from torch.functional import F
import util



def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


class AutoRebalanceEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_price: pd.DataFrame, targets, means, covs, re_target, alpha=3.35, len_period: int = 500,
                 mode='train', start_tick=0, T= 500):
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
        self.T = T
        self.mu = None
        self.cov = None
        self.target_ratio = None
        self.cum_target = None
        self.re_target = re_target
        self.daily_growth = util.daily_growth(self.stock_price)

        # step variable
        self.done = False
        self.start_tick = None
        self.current_tick = None
        self.last_tick = None
        self.target_ratio = None
        self.cum_target = None

        self.current_weight = None
        self.current_return = None
        self.growth_rate = None
        self.n_train = -1
        self.n_rebalance = None
        self.total_cost = None
        self.total_error = None
        self.total_utility_error = None

        self.history = None

        # action and observation space
        self.n_actions = self.stock_price.shape[1]
        # self.action_space = spaces.Box(low=self.cum_target - .05, high=self.cum_target + .05,
        #                                shape=(self.n_actions - 1,), dtype=np.float64)
        self.action_space = spaces.Box(low=0.01, high=1, shape=(self.n_actions, ), dtype=np.float64)
        self.observation_space = spaces.Box(low=-3, high=1, shape=(self.n_actions*2, self.T, ), dtype=np.float64)

    def step(self, action: np.ndarray):
        action = softmax(action)

        if self.current_weight is not None:
            trading_cost = self._get_trading_cost(self.current_weight, action)
            if np.sum(np.abs(action - self.current_weight)) >= .01:
                self.n_rebalance += 1
        else:
            trading_cost = 0

        self.current_weight = action

        self.current_weight *= self.daily_growth[self.current_tick]
        current_return = np.sum(self.current_weight)
        self.current_weight /= current_return
        self.growth_rate *= current_return

        self.current_tick += 1
        log_return = np.log(self.daily_growth[self.current_tick - self.T:self.current_tick])
        target_ratios = self.targets[self.current_tick+1 - self.T:self.current_tick+1]
        observation = np.concatenate([log_return, target_ratios], axis=1).T

        reward = np.log(current_return - trading_cost)
        self.total_cost += trading_cost
        self.done = self.current_tick == self.last_tick

        info = {
            'Observation': observation,
            'Current Weight': self.current_weight,
            'Action': action,
            '# Rebalance': self.n_rebalance,
            'Trading Cost': trading_cost,
            'Daily Return': self.current_return,
            'Total Cost': self.total_cost,
            'Total Return': self.growth_rate,
            'Reward': reward,
        }
        self._update_history(info)
        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self.n_train += 1
        self.n_rebalance = 0

        if self.MODE == 'train' and (self.n_train % self.re_target) == 0:
            self.start_tick = np.random.choice(range(self.T, len(self.stock_price) - self.LEN_OF_PERIOD))
        else:
            self.start_tick = self.T
        self.current_tick = self.start_tick
        self.last_tick = self.current_tick + self.LEN_OF_PERIOD - 1
        self.current_weight = None
        self.growth_rate = 1.
        self.history = None

        self.total_cost = 0
        log_return = np.log(self.daily_growth[self.current_tick - self.T:self.current_tick])
        target_ratios = self.targets[self.current_tick+1 - self.T:self.current_tick+1]
        observation = np.concatenate([log_return, target_ratios], axis=1).T


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

    def _get_trading_cost(self, w, w_goal):
        cost = 0
        for a in (w_goal-w):
            if a < 0:
                cost += -a * self.SELL_COST ## a < 0 -> -a
            else:
                cost += a * self.BUY_COST
        # tc *= self.growth_rate
        return cost

    def update_target(self):
        if self.n_step % self.re_target == 0:
            self.target_ratio = self.targets[self.current_tick]

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
    start_day = '2017-01-01'
    end_day = '2019-01-01'
    re_target = 50
    T = 500
    df = pd.read_csv("data/1992_2019_daily_data_with_target.csv", index_col='Dates')
    means = np.load("data/means.npy")
    covs = np.load("data/covs.npy")

    train_df, test_df = df[:start_day], df[start_day:end_day]
    len_period = len(test_df)

    train_df, test_df = train_df.iloc[:-T], pd.concat([train_df.iloc[:T], test_df])
    test_price = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    test_targets = test_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    test_means = means[len(train_df):]
    test_covs = covs[len(train_df):]
    test_env = AutoRebalanceEnv(alpha=3.35, stock_price=test_price, targets=test_targets, means=test_means,
                                covs=test_covs, len_period=len_period, re_target=re_target, mode='test', T=T)

    obs = test_env.reset()
    while True:
        action = test_env.action_space.sample()
        obs, rewards, dones, info = test_env.step(action)
        if dones:
            break
