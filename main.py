# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C, TD3, DDPG, PPO, SAC
from sklearn.model_selection import train_test_split
from env import AutoRebalanceEnv
import pandas as pd
import numpy as np
import gym
import util
# gym.logger.set_level(40)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # len_period = 500
    # df = pd.read_csv('2000_2019_daily_data.csv')[['H0A0 Index', 'NKY Index', 'SENSEX Index', 'SPX Index']].to_numpy()
    # train_df = df[:-len_period]
    # test_df = df[-len_period:]
    # target = np.array([.13, .14, .17, .56])
    train_df1 = pd.read_csv('data/1990_1999_daily_data.csv')
    train_df1 = train_df1[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    train_df = pd.read_csv('data/2000_2009_daily_data.csv')
    train_df = train_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    # train_df = train_df[['SHCOMP Index', 'HSCEI Index',  'SENSEX Index', 'MXLA Index']].to_numpy()
    test_df = pd.read_csv('data/2010_2019_daily_data.csv')
    test_df = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    train_df = np.concatenate([train_df1, train_df])
    train_df = np.concatenate([train_df1, train_df, test_df[:1828]])
    test_df = test_df
    Dates = pd.read_csv('data/2010_2019_daily_data.csv')['Dates']
    start_day = Dates[1828]
    end_day = Dates[len(Dates)-1]
    # test_df = test_df[['SHCOMP Index', 'HSCEI Index',  'SENSEX Index', 'MXLA Index']].to_numpy()
    # target = np.array([.1164, .2988, .3090, .2758])
    target = np.array([.4405, .2694, .1375, .1526])
    len_period = len(test_df) - 1828
    daily_growth = util.daily_growth(test_df)
    train_env = AutoRebalanceEnv(alpha=3.35, stock_price=train_df, len_period=len_period,
                                 target_ratio=target)
    test_env = AutoRebalanceEnv(alpha=3.35, stock_price=test_df, len_period=len_period,
                                target_ratio=target, mode='test', start_tick=1828)
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
    test_env.save_history(f'a2c_return-{history["Total Return"][-1]*100:.2f}_sharpe-{util.sharpe_ratio(history["Total Return"]):.4f}_cost-{history["Total Cost"][-1]*100:.2f}%_count-{history["# Rebalance"][-1]}_len-{len_period}.csv')
    with open('results.csv', 'a') as f:
        f.write(f'a2c, {history["Total Return"][-1]*100:.2f}, {util.sharpe_ratio(history["Total Return"]):.4f}, {history["Total Cost"][-1]*100:.2f}, {history["# Rebalance"][-1]}, {start_day}, {end_day}\n')

    util.get_buy_n_hold_return(target, daily_growth, len_period)
