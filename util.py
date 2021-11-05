import numpy as np
import pandas as pd
import cvxpy as cp

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
    return growth



def tolerance(targets, daily_growth, tol_rate):
    cost = 0
    count = 0
    growth = np.zeros(len(daily_growth))
    current_weight = targets[0].copy()
    total_growth = 1
    for i, g in enumerate(daily_growth):
        ratio = targets[i]
        current_weight *= (1 + g)
        total_growth *= np.sum(current_weight)
        current_weight = norm(current_weight)
        growth[i] = total_growth
        if np.sum(abs(ratio - current_weight)) >= tol_rate:
            count += 1
            for d in ratio - current_weight:
                if d > 0:
                    cost += .001415 * d * total_growth
                else:
                    cost -= .004415 * d * total_growth
            current_weight = ratio.copy()

    return growth, cost, count

def PR(targets, daily_growth, period):
    cost = 0
    count = 0
    days = 0
    growth = np.zeros(len(daily_growth))
    current_weight = targets[0].copy()
    total_growth = 1
    for i, g in enumerate(daily_growth):
        ratio = targets[i]
        days += 1
        current_weight *= (1 + g)
        total_growth *= np.sum(current_weight)
        current_weight = norm(current_weight)
        growth[i] = total_growth
        if days % period == 0:
            count += 1
            for d in ratio - current_weight:
                if d > 0:
                    cost += .001415 * d * total_growth
                else:
                    cost -= .004415 * d * total_growth
            current_weight = ratio.copy()

    return growth, cost, count


def sharpe_ratio(growth):
    daily_growth = []
    for i, g in enumerate(growth[:-1]):
        daily_growth.append(growth[i+1]/g - 1)
    return np.mean(daily_growth)/np.std(daily_growth)


if __name__ == '__main__':
    df = pd.read_csv("data/1992_2019_daily_data_with_target.csv", index_col='Dates')
    start_day = "2017-10-29"
    end_day = "2019-10-29"
    test_df = df[start_day:end_day]
    test_price = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    targets = test_df[['tg1', 'tg2', 'tg3', 'tg4']].to_numpy()
    daily_growth = daily_growth(test_price)



    for tol_rate in range(5):
        tol_rate = .01 * (tol_rate + 1)
        growth, cost, count = tolerance(targets, daily_growth, tol_rate)
        with open(f'{start_day}_{end_day}_results.csv', 'a') as f:
            f.write(f'TR({tol_rate*100:d}%), {growth[-1]*100:.2f}, {sharpe_ratio(growth):.4f}, {cost*100:.2f}, {count}, {start_day}, {end_day}\n')

    for period in [1, 2, 3, 4, 5, 6, 9, 12]:
        growth, cost, count = PR(targets, daily_growth, period)
        with open(f'{start_day}_{end_day}_results.csv', 'a') as f:
            f.write(f'PR({period}), {growth[-1]*100:.2f}, {sharpe_ratio(growth):.4f}, {cost*100:.2f}, {count}, {start_day}, {end_day}\n')

    growth = get_buy_n_hold_return(targets[0], daily_growth, len(test_df))
    with open(f'{start_day}_{end_day}_results.csv', 'a') as f:

        f.write(f'BH, {growth[-1] * 100:.2f}, {sharpe_ratio(growth):.4f}, {0}, {1}, {start_day}, {end_day}\n')

