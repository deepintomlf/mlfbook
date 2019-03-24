from scipy.cluster.hierarchy import dendrogram, linkage
import pylab
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(eventside_only=True):
    p = '/content/drive/My Drive/Colab Notebooks/data/'
    msg = pd.read_csv(f'{p}/message.csv',header=None)
    msg.columns = ['Time', 'Type', 'Order ID', 'Size', 'Price', 'Direction']
    data = pd.read_csv(f'{p}/orderbook.csv', header=None)
    data.columns = chain.from_iterable(
        [[f'ask_price_{i}', f'ask_size_{i}', f'bid_price_{i}', f'bid_size_{i}'] for i in range(1, 6)])
    d = pd.concat([msg, data], axis=1)
    columns_price = d.columns[d.columns.str.contains('price', regex=False, case=False)]
    d[columns_price] = d[columns_price] / 1e4
    res = d
    if eventside_only:
        dfs = []
        for direction, df in d.groupby('Direction'):
            attr_side = 'bid_' if direction == 1 else 'ask_'
            cols = d.columns[d.columns.str.contains(attr_side, regex=False, case=False)]
            cols_new = [s.replace(attr_side, '') for s in cols]
            duni = df[list(msg.columns) + list(cols)].rename(columns=dict(zip(cols, cols_new)))
            dfs.append(duni)
        res = pd.concat(dfs)
    return res



