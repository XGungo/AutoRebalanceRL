# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C, TD3, DDPG, PPO, SAC
from sklearn.model_selection import train_test_split
from env import AutoRebalanceEnv
import pandas as pd
import numpy as np
import sys
import util

# gym.logger.set_level(40)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        re_target = int(sys.argv[1])
    except IndexError:
        re_target = 1
    print(f'n_target:{re_target}')
    df = pd.read_csv("data/1992_2019_daily_data_with_target.csv", index_col='Dates')
    means = np.load("data/means.npy")
    covs = np.load("data/covs.npy")
    start_day = "2017-10-29"
    end_day = "2019-10-29"
    train_df, test_df = df[:start_day], df[start_day:end_day]

    train_price = train_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    train_targets = train_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    test_price = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    test_targets = test_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    train_means, test_means = means[:len(train_df)], means[len(train_df):]
    train_covs, test_covs = covs[:len(train_df)], covs[len(train_df):]
    len_period = len(test_df)

    train_env = AutoRebalanceEnv(alpha=3.35, stock_price=train_price, targets=train_targets, means=train_means,
                                 covs=train_covs, len_period=len_period, re_target=re_target)
    test_env = AutoRebalanceEnv(alpha=3.35, stock_price=test_price, targets=test_targets, means=test_means,
                                covs=test_covs, len_period=len_period, re_target=re_target, mode='test')
    model = A2C("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=250000)
    obs = test_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        if dones:
            break
        test_env.render()
    history = test_env.history

    # test_env.save_history(f'a2c_return-{history["Total Return"][-1] * 100:.2f}_' +
    #                       'sharpe-{util.sharpe_ratio(history["Total Return"]):.4f}_' +
    #                       'cost-{history["Total Cost"][-1]*100:.2f}%_' +
    #                       'count-{history["# Rebalance"][-1]}_len-{len_period}.csv')

    with open('results.csv', 'a') as f:
        f.write(f'a2c_{re_target}, {history["Total Return"][-1] * 100:.2f}, {util.sharpe_ratio(history["Total Return"]):.4f}, ' +
                f'{history["Total Cost"][-1] * 100:.2f}, {history["# Rebalance"][-1]}, {start_day}, {end_day}\n')
