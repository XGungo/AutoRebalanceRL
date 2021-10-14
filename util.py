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



def tolerance(ratio, daily_growth, tol_rate):
    cost = 0
    count = 0
    growth = np.zeros(len(daily_growth))
    current_weight = ratio.copy()
    total_growth = 1
    for i, g in enumerate(daily_growth):
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

def PR(ratio, daily_growth, period):
    cost = 0
    count = 0
    days = 0
    growth = np.zeros(len(daily_growth))
    current_weight = ratio.copy()
    total_growth = 1
    for i, g in enumerate(daily_growth):
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


class Portolio:
    def __init__(self, mu, cov, alpha=1):
        self.mu = mu
        self.cov = cov
        self.alpha = alpha

    def get_target_ratio(self):
        X = cp.Variable(self.mu.shape[0])
        constraints = [cp.sum(X) == 1]
        constraints += [X>=0, X <= 1]
        objective = cp.Maximize(X.T@self.mu - self.alpha/2*cp.sum(X**2@self.cov))
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print(X.value)


if __name__ == '__main__':
    test_df = pd.read_csv('2010_2019_daily_data.csv')
    target = np.array([.4405, .2694, .1375, .1526])
    test_df = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    test_df = test_df[1828:]

    daily_growth = daily_growth(test_df)
    #
    # for tol_rate in range(5):
    #     tol_rate = .01 * (tol_rate + 1)
    #     growth, cost, count = tolerance(target, daily_growth, tol_rate)
    #     np.savetxt(f'tolerance_{tol_rate*100}%_return-{growth[-1]*100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_cost-{cost*100:.2f}%_count-{count}_len-{len(test_df)}.csv', growth)
    #
    # for period in [1, 2, 3, 4, 5, 6, 9, 12]:
    #     growth, cost, count = PR(target, daily_growth, period)
    #     np.savetxt(f'PR({period})_return-{growth[-1]*100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_cost-{cost*100:.2f}%_count-{count}_len-{len(test_df)}.csv', growth)
    # # growth = get_buy_n_hold_return(target, daily_growth, len(test_df))
    # # np.savetxt(f'BH_return-{growth[-1] * 100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_len-{len(test_df)}.csv', growth)
    portfolio = Portolio(mu=np.mean(daily_growth, axis=0), cov=np.cov(daily_growth.T))
