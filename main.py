# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C
from sklearn.model_selection import train_test_split
from env import AutoRebalanceEnv
import pandas as pd
import numpy as np
import gym
import util
# gym.logger.set_level(40)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    len_period = 500
    df = pd.read_csv('2000_2019_daily_data.csv')[['H0A0 Index', 'NKY Index', 'SENSEX Index', 'SPX Index']].to_numpy()
    train_df = df[: -len_period]
    test_df = df[-len_period:]
    target = np.array([.13, .14, .17, .56])
    daily_growth = util.daily_growth(test_df)

    train_env = AutoRebalanceEnv(alpha=3.35, stock_price=train_df, len_period=len_period,
                                 target_ratio=np.array([.13, .14, .17, .56]))
    test_env = AutoRebalanceEnv(alpha=3.35, stock_price=test_df, len_period=len_period,
                                target_ratio=np.array([.13, .14, .17, .56]))
    model = A2C("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=250000)
    model.save("a2c_autoRebalance")
    obs = test_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        if dones:
            break
        test_env.render()
    test_env.save_history(f'a2c_autoRebalance_{len_period}.csv')
    util.get_buy_n_hold_return(target, daily_growth, len_period)
