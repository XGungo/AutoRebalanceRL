import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(12,8))
plt.title(r'Return')
plt.xlabel("進場時間")
plt.ylabel("Percentage(%)")
p = []
counter = 0
for file in ['a2c_results.csv']:
    df = pd.read_csv(file)
    Returns = []
    means = []
    x_axis = []
    
    other = pd.DataFrame(columns=['model','return','sharpe','cost','count','start','end'])
    for group,data in df.groupby('start'):
        Returns.append(data['return'] - data['cost'])
        means.append(np.mean(data['return'] - data['cost']))
        x_axis.append(group)
    boxpt = plt.boxplot(Returns)
    line, = plt.plot(range(1, len(means)+1), means, label='A2C')
    p.append(line)
    plt.xticks(range(1, len(x_axis)+1), x_axis, rotation='vertical')

# plt.show()
plt.savefig('return.png')
