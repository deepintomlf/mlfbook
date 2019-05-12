from scipy.cluster.hierarchy import dendrogram, linkage
import pylab
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

def get_tick_size(orders):
    px_change = lambda orders, ptype: [x for x in sorted(orders[ptype].diff().abs().unique()) if x > 0]
    min_ab = lambda a, b: min(min(a), min(b))
    tick_size = np.round(min_ab(px_change(orders, 'askPrice'), px_change(orders, 'bidPrice')), 2)
    return tick_size


def orders_execution_lit_enriched(data):
    orders = data[data['type'] == 4].sort_values(by='time').reset_index(drop=True)
    tick_size = get_tick_size(orders)
    orders['smid_quote'] = (orders['askPrice'] + orders['bidPrice']) / 2
    orders['spread_dollar'] = orders['askPrice'] - orders['bidPrice']
    orders['spread_bps'] = orders['spread_dollar'] / orders['smid_quote'] * 1e4
    orders['spread_tick'] = orders['spread_dollar'] / tick_size
    orders['return_smid_quote_log'] = np.log(orders['smid_quote']) - np.log(orders['smid_quote'].shift(1))
    orders['depth'] = orders['askSize'] + orders['bidSize']
    orders['depth_log'] = np.log(orders['askSize']) + np.log(orders['bidSize'])
    orders['depth_dollar'] = (orders['askPrice'] * orders['askSize'] + orders['bidPrice'] * orders['bidSize']) / 2
    orders['imbalance'] = (orders['bidSize'] - orders['askSize']) / (orders['bidSize'] + orders['askSize'])
    orders['imbalance_bucket'] = pd.cut(orders['imbalance'], np.arange(-1, 1.01, 0.2))
    return orders





