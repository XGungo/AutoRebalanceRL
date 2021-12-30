import os
import pandas as pd
from multiprocessing import Process, Pool

def training(start_end):
    os.system(f"python main.py {start_end[0]} {start_end[1]}")
    return f"{start_end[0]} {start_end[1]} done."

if __name__ == '__main__':
    dates = pd.date_range('2015-01-01', '2019-01-01', freq='1M') - pd.offsets.MonthBegin(1)
    dates = pd.to_datetime(dates, format='%Y-%m-%d').strftime('%Y-%m-%d')
    with open(f'a2c_results.csv', 'a') as f:
        f.write('model,return,sharpe,cost,count,start,end\n')
    start_end = []
    for i in range(len(dates) - 24):
        start_end.append((dates[i], dates[i + 24]))

    pool = Pool(3)
    pool_outputs = pool.map(training, start_end)



