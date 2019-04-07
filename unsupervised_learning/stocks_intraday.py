from scipy.cluster.hierarchy import dendrogram, linkage
import pylab
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_ob_data_ml(eventside_only=True):
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


def load_ob_data_toplvl(symbol,date,base_dir='/content/drive/My Drive/2019WritingBook/data/hft/'):
    [fn_msg, fn_ob] = [f'{base_dir}/{symbol.upper()}/{symbol.upper()}_{date}_{ftype}.csv' for ftype in
                       ['message', 'orderbook']]
    msg = pd.read_csv(fn_msg, header=None).loc[:, 0:5]
    msg.columns = ['time', 'type', 'orderId', 'size', 'price', 'direction']
    ob = pd.read_csv(fn_ob, header=None)
    ob.columns = ['askPrice', 'askSize', 'bidPrice', 'bidSize']
    d = pd.concat([msg, ob], axis=1)
    columns_price = d.columns[d.columns.str.contains('price', regex=False, case=False)]
    d[columns_price] = d[columns_price] / 1e4
    return d

def orders_execution_lit_enriched(data):
    orders = data[data['type'] == 4].sort_values(by='time').reset_index(drop=True)
    px_change = lambda orders, ptype: [x for x in sorted(orders[ptype].diff().abs().unique()) if x > 0]
    min_ab = lambda a, b: min(min(a), min(b))
    tick_size = np.round(min_ab(px_change(orders, 'askPrice'), px_change(orders, 'bidPrice')), 2)
    orders['smid_quote'] = (orders['askPrice'] + orders['bidPrice']) / 2
    orders['spread_dollar'] = orders['askPrice'] - orders['bidPrice']
    orders['spread_bps'] = orders['spread_dollar'] / orders['smid_quote'] * 1e4
    orders['spread_tick'] = orders['spread_dollar'] / tick_size
    orders['return_smid_quote_log'] = np.log(orders['smid_quote']) - np.log(orders['smid_quote'].shift(1))
    orders['depth'] = orders['askSize'] + orders['bidSize']
    orders['depth_log'] = np.log(orders['askSize']) + np.log(orders['bidSize'])
    orders['depth_dollar'] = (orders['askPrice'] * orders['askSize'] + orders['bidPrice'] * orders['bidSize']) / 2
    return orders





