import os
import pandas as pd

dates = pd.date_range('2000-01-01', '2019-10-01', freq='1M') - pd.offsets.MonthBegin(1)
dates = pd.to_datetime(dates, format='%Y-%m-%d').strftime('%Y-%m-%d')
with open(f'a2c_results.csv', 'a') as f:
    f.write('model,return,sharpe,cost,count,start,end\n')

for i in range(len(dates) - 24):
    start_day = dates[i]
    end_day = dates[i+24]
    os.system(f"python main.py {start_day} {end_day}")

