from scipy.cluster.hierarchy import dendrogram, linkage
import pylab
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dendrogram(data,linkage_type='average'):
    Z = linkage(data, linkage_type)
    plt.figure(figsize=(20, 10))
    labelsize = 20
    plt.title('Hierarchical Clustering Dendrogram', fontsize=labelsize)
    plt.xlabel('stock', fontsize=labelsize)
    plt.ylabel('distance', fontsize=labelsize)
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        labels=data.columns
    )

def load_data():
    p = '/content/drive/My Drive/Colab Notebooks/data/'
    data = pd.read_csv('{p}/SX5E_daily_2018.csv'.format(p=p), header=None)
    d = data.iloc[6:, :]
    d.columns = data.iloc[5, :]
    d = d.set_index('Dates')
    names = data.iloc[3, :].dropna().unique()
    features = d.columns.unique()
    d.columns = pd.MultiIndex.from_product([names, features], names=['symbol', 'features'])
    d = d.dropna().astype(float)
    return d

def data_features(d):
    idx = pd.IndexSlice
    px_close = d.loc[:, idx[:, 'PX_LAST']]
    px_open = d.loc[:, idx[:, 'PX_OPEN']]
    px_volume = d.loc[:, idx[:, 'PX_VOLUME']] / 1e3
    px_return = (d.loc[:, idx[:, 'PX_LAST']].diff().dropna() / d.loc[:, idx[:, 'PX_LAST']].iloc[1:, :])
    vwap_volume = d.loc[:, idx[:, 'VWAP_VOLUME']] / 1e3
    for x in [px_close, px_open, px_return, px_volume, vwap_volume]:
        x.columns = x.columns.droplevel(1)
    px_variation = px_close - px_open
    px_overnight = px_open.iloc[1:, :] - px_close.iloc[0:-1, :]
    # agg on dates
    normalize = lambda df: df.mean().div(df.std())
    mean = lambda df: df.mean()
    fvolume = px_volume.mean()
    fvariation = px_variation.mean() / px_variation.std(axis=0)
    fopen = normalize(px_open)
    fclose = normalize(px_close)
    freturn = mean(px_return)
    fovernight = mean(px_overnight)
    features_map = {'open': fopen,
                   'close': fclose,
                   'volume': fvolume,
                   'variation': fvariation,
                   'return': freturn,
                   'overnight': fovernight}
    return features_map



def cluster_features():
    pass


def cluster_names():
    pass


def cluster_with_pca():
    pass

