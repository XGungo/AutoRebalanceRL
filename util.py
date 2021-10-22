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
        objective = cp.Minimize(X.T@self.mu - self.alpha/2*X.T@self.cov@X)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.SCS)
        print(X.value)

class Convex:
    def __init__(self, mu, cov, lr=.2, alpha=1):
        self.mu = mu
        self.cov = cov
        self.alpha = alpha
        self.lr = lr
    def utility(self, w):
        return w.T@self.mu - self.alpha/2*w.T@self.cov@w

    @staticmethod
    def x_to_w(x):
        cum_w = np.zeros(len(x)+2)
        cum_w[1:-1] = np.sort(x)
        cum_w[-1] = 1
        w = np.diff(cum_w)
        return w

    @staticmethod
    def check_x(x):
        if np.sum(x != np.sort(x)) > 0 or x[0] < 0 or x[-1] > 1:
            return 0
        else:
            return 1

    def get_best(self):
        x = np.sort(np.random.uniform(0, 1, 3))
        for _ in range(10**4):
            new_x = x
            for idx in range(len(x)):
                h = np.random.normal(0, .1)
                temp_x = x
                temp_x[idx] += h
                gradient = (self.utility(Convex.x_to_w(temp_x)) - self.utility(Convex.x_to_w(x)))/h
                new_x[idx] -= self.lr * gradient
                if not Convex.check_x(new_x):
                    new_x[idx] += self.lr*gradient
            x = new_x
        return self.x_to_w(x)




if __name__ == '__main__':
    test_df = pd.read_csv('raw data/2010_2019_daily_data.csv')
    target = np.array([.4405, .2694, .1375, .1526])
    test_df = test_df[['SPX Index', 'SHCOMP Index', 'SENSEX Index', 'MXLA Index']].to_numpy()
    test_df = test_df[512:]
    Dates = pd.read_csv('raw data/2010_2019_daily_data.csv')['Dates']
    start_day = Dates[512]
    end_day = Dates[len(Dates)-1]

    daily_growth = daily_growth(test_df)

    for tol_rate in range(5):
        tol_rate = .01 * (tol_rate + 1)
        growth, cost, count = tolerance(target, daily_growth, tol_rate)
        with open('results.csv', 'a') as f:
            f.write(f'tolerance{tol_rate*100}%, {growth[-1]*100:.2f}, {sharpe_ratio(growth):.4f}, {cost*100:.2f}, {count}, {start_day}, {end_day}\n')
        # np.savetxt(f'tolerance_{tol_rate*100}%_return-{growth[-1]*100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_cost-{cost*100:.2f}%_count-{count}_len-{len(test_df)}.csv', growth)

    for period in [1, 2, 3, 4, 5, 6, 9, 12]:
        growth, cost, count = PR(target, daily_growth, period)
        with open('results.csv', 'a') as f:
            f.write(f'PR({period}), {growth[-1]*100:.2f}, {sharpe_ratio(growth):.4f}, {cost*100:.2f}, {count}, {start_day}, {end_day}\n')
        # np.savetxt(f'PR({period})_return-{growth[-1]*100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_cost-{cost*100:.2f}%_count-{count}_len-{len(test_df)}.csv', growth)
    growth = get_buy_n_hold_return(target, daily_growth, len(test_df))
    with open('results.csv', 'a') as f:
        f.write(f'Buy and Hold, {growth[-1] * 100:.2f}, {sharpe_ratio(growth):.4f}, {0}, {1}, {start_day}, {end_day}\n')
    # np.savetxt(f'BH_return-{growth[-1] * 100:.2f}_sharpe-{sharpe_ratio(growth):.4f}_len-{len(test_df)}.csv', growth)
    # portfolio = Portolio(mu=np.mean(daily_growth, axis=0), cov=np.cov(daily_growth.T))
    # convex = Convex(mu=np.mean(daily_growth, axis=0), cov=np.cov(daily_growth.T))
