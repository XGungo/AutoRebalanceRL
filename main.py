# This is a sample Python script.

import sys

import numpy as np
import pandas as pd
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
import util
from env import AutoRebalanceEnv
from policy import CustomCNN, CustomActorCriticPolicy
# gym.logger.set_level(40)

# Press the green button in the gutter to run the script.

log_dir = "./a2c_auto_rebalance/"
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)

        return True
    def _on_rollout_end(self) -> None:
        value = self.training_env.get_attr("growth_rate")[0]
        cost = self.training_env.get_attr("total_cost")[0]
        self.logger.record('Return', value)
        self.logger.record('Cost', cost)
        self.logger.record('Net Return', value - cost)



if __name__ == '__main__':
    try:
        start_day = sys.argv[1]
        end_day = sys.argv[2]
    except IndexError:
        start_day = '2017-01-01'
        end_day = '2019-01-01'
    print(start_day, end_day)
    re_target = 100
    T = 500
    df = pd.read_csv("data/1992_2019_daily_data_with_target.csv", index_col='Dates')
    means = np.load("data/means.npy")
    covs = np.load("data/covs.npy")



    train_df, test_df = df[:start_day], df[start_day:end_day]
    len_period = len(test_df)
    test_df = pd.concat([train_df.iloc[-T:], test_df])


    train_price = train_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    train_targets = train_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    test_price = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    test_targets = test_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    train_means, test_means = means[:len(train_df)], means[-len(test_df):]
    train_covs, test_covs = covs[:len(train_df)], covs[-len(test_df):]

    train_env = AutoRebalanceEnv(stock_price=train_price, targets=train_targets, means=train_means,
                                 covs=train_covs, len_period=len_period, re_target=re_target, T=T)
    test_env = AutoRebalanceEnv(stock_price=test_price, targets=test_targets, means=test_means,
                                covs=test_covs, len_period=len_period, re_target=re_target, mode='test', T=T)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )
    model = A2C('MlpPolicy', train_env, verbose=1,  tensorboard_log=log_dir, policy_kwargs=policy_kwargs, )
    model.learn(total_timesteps=int(4e6), callback=TensorboardCallback(), tb_log_name=f"{start_day}-{end_day}")
    del train_env
    for _ in range(100):
        obs = test_env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = test_env.step(action)
            if dones:
                break
            # test_env.render()
        history = test_env.history

        with open(f'a2c_results.csv', 'a') as f:
            f.write(f'a2c, {history["Total Return"][-1] * 100:.2f},' +
                    f'{util.sharpe_ratio(history["Total Return"]):.4f}, ' +
                    f'{history["Total Cost"][-1] * 100:.2f}, {history["# Rebalance"][-1]}, {start_day}, {end_day}\n')
