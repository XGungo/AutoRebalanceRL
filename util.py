import numpy as np
import pandas as pd


def daily_growth(daily_stock_price, mode='monthly'):
    diff = np.diff(daily_stock_price, axis=0)
    return diff / daily_stock_price[:-1]


def expected_mean_return(daily_stock_price):
    daily_return_ = daily_growth(daily_stock_price)
    return np.mean(daily_return_, axis=0)


def norm(vector: np.ndarray):
    return vector / np.sum(vector)


def xlsx_to_csv(filename):
    data_xls = pd.read_excel(f'{filename}.xlsx', index_col=0)
    data_xls.replace(0, np.nan, inplace=True)
    data_xls = data_xls.dropna(axis=1)
    data_xls.to_csv(f'./{filename}', encoding='utf-8')


def get_buy_n_hold_return(target, daily_growth, period):
    current_weight = target
    growth = np.zeros(len(daily_growth))
    for i, g in enumerate(daily_growth):
        current_weight *= (1 + g)
        growth[i] = np.sum(current_weight)
    np.savetxt(f'buy_n_hold_growth_{period}.csv', growth)


if __name__ == '__main__':
    xlsx_to_csv('1995_2019 daily data')
