from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from math import ceil
from sklearn.metrics.pairwise import pairwise_distances_argmin

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from scipy.cluster.hierarchy import dendrogram, linkage
import pylab

class ClusteringAnalysis(object):
    def __init__(self, data=None, true_labels=None):
        self.data = data
        self.labels = true_labels

    def load_data_iris(self, features=None, true_label_name=None):
        data = sns.load_dataset("iris")
        X = data.loc[:, features] if features else data
        true_labels = data.loc[:, true_label_name] if true_label_name in data.columns else []
        return X, true_labels

    def model_kmeans(self, data, n_clusters=3, init_method='k-means++', n_init=10):
        model = KMeans(n_clusters=n_clusters,
                       init=init_method,
                       n_init=n_init)
        model.fit(data)
        cluster_labels = model.predict(data)
        cluster_centers = model.cluster_centers_
        fit_res = {'cluster_method': 'kmeans',
                   'model': model,
                   'cluster_labels': cluster_labels,
                   'cluster_centers': cluster_centers,
                   'data': data}
        return fit_res


    def model_DBSCAN(self,data,epsilon=None,min_pts=None):
        epsilon = 0.3 if epsilon is None else epsilon
        min_pts = data.shape[1] if min_pts is None else min_pts
        db = DBSCAN(eps=epsilon,min_samples=min_pts).fit(data)
        fit_res = {
            'cluster_method': 'DBSCAN',
            'model':'db',
            'cluster_labels': db.labels_,
            'data': data
        }
        return fit_res


    def plot_dendrogram(self,data):
        Z = linkage(data, 'average')
        fig = plt.figure(figsize=(25, 10))
        labelsize = 20
        ticksize = 15
        plt.title('Hierarchical Clustering Dendrogram', fontsize=labelsize)
        plt.xlabel('stock', fontsize=labelsize)
        plt.ylabel('distance', fontsize=labelsize)
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            labels=data.index
        )
        pylab.yticks(fontsize=ticksize)
        pylab.xticks(rotation=-90, fontsize=ticksize)
        return fig

    def plot_scatter_kmeans(self, X, paramtype, params, feature_pair, n_clusters=3):

        if paramtype == 'n_iteration':
            colors = ['orange', 'lightblue', 'green']
            niters = params
            nplots = len(params)
            ncols = 2
            nrows = ceil(nplots / ncols)
            [h_col, h_row] = [5, 5]
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * h_col, nrows * h_row))
            for (niter, ax) in zip(niters, axes.flatten()):
                # ------------- fit X with Kmeans => cluster labels and cluster centers -------- #
                kmeans = self.model_kmeans(data=X, n_clusters=n_clusters, rs=0, n_init=1, max_iter=niter)
                cluster_labels = kmeans['cluster_labels']
                cluster_centers = kmeans['cluster_centers']
                mode_fitted = kmeans['model']
                niter = min(mode_fitted.n_iter_, niter)
                # ------------- plot clusters for each iteration -------- #
                # cluster labels can change in different runs, to keep the same color for the "same" cluster, we make an order by pairing cluster center
                # with the first one, then we use the order for color_map
                if niter == niters[0]:
                    cluster_centers_init = cluster_centers.copy()
                label_order = pairwise_distances_argmin(cluster_centers, cluster_centers_init) if niter != niters[
                    0] else sorted(list(set(cluster_labels)))
                color_map = dict(zip(label_order, colors))
                ax.scatter(X[feature_pair[0]], X[feature_pair[1]], c=[color_map[x] for x in cluster_labels], alpha=0.6)
                ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
                # add cluster center annotion
                for xy in tuple(np.round(cluster_centers, 2)):
                    ax.annotate("({},{})".format(xy[0], xy[1]), xy=(xy[0], xy[1]))
                ax.set_title('KMeans: after {} iterations'.format(niter))
                ax.set_xlabel(feature_pair[0])
                ax.set_ylabel(feature_pair[1])
            fig.tight_layout()

        elif paramtype == 'n_clusters':
            nplots = len(params)
            ncols = 2
            nrows = ceil(nplots / ncols)
            [h_col, h_row] = [5, 5]
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * h_col, nrows * h_row))
            for (param, ax) in zip(params, axes.flatten()):
                # ------------- fit X with Kmeans => cluster labels and cluster centers -------- #
                kmeans = self.model_kmeans(data=X, n_clusters=param, rs=0, n_init=1)
                cluster_labels = kmeans['cluster_labels']
                cluster_centers = kmeans['cluster_centers']
                mode_fitted = kmeans['model']
                label_order = sorted(list(set(cluster_labels)))
                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, param)]
                color_map = dict(zip(label_order, colors))
                ax.scatter(X[feature_pair[0]], X[feature_pair[1]], c=[color_map[x] for x in cluster_labels], alpha=0.8)
                ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='*')
                # add cluster center annotion
                for xy in tuple(np.round(cluster_centers, 2)):
                    ax.annotate("({},{})".format(xy[0], xy[1]), xy=(xy[0], xy[1]))
                ax.set_title('KMeans: number of clusters = {}'.format(param))
                ax.set_xlabel(feature_pair[0])
                ax.set_ylabel(feature_pair[1])
            fig.tight_layout()
        return fig


    def metric_silhouette_width(self, X, cluster_labels):
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        silhouette_sample = silhouette_samples(X, cluster_labels)
        n_clusters = len(set(cluster_labels))
        res = {'silhouette_avg': silhouette_avg, 'silhouette_sample': silhouette_sample, n_clusters: 'n_clusters'}
        return res

    def plot_silhouette_width(self, fit_res):
        X = fit_res['data']
        cluster_labels = fit_res['cluster_labels']
        n_clusters = len(set(cluster_labels))
        cluster_centers = fit_res['cluster_centers']
        colors = iter(cm.cividis(np.linspace(0, 1, n_clusters)))
        color_map = {}
        fig, (ax_sh, ax_cluster) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        xcluster = X.copy()
        xcluster['cluster_label'] = cluster_labels
        sw = self.metric_silhouette_width(X, cluster_labels)
        silhouette_avg = sw['silhouette_avg']
        xcluster['silhouette_sample'] = sw['silhouette_sample']
        cluster_steps = [10]
        for label, df in xcluster.groupby('cluster_label'):
            cluster_size = df.shape[0]
            # choose color, cluster band location
            cluster_steps.append(cluster_steps[-1] + cluster_size)
            c = next(colors)
            color_map.update({label: c})
            # sort by silhouette_value
            sh_sorted = np.array(sorted(df['silhouette_sample'].values))
            ax_sh.fill_betweenx(np.arange(cluster_steps[-2], cluster_steps[-1]), 0, sh_sorted,
                                facecolor=c, edgecolor=c, alpha=0.6)
            ax_sh.text(-0.1, cluster_steps[-2] + 0.5 * cluster_size, str(label))
            vl = ax_sh.axvline(x=silhouette_avg, color="red", linestyle=":", alpha=0.8)
        ax_sh.set_yticks([])
        ax_sh.set_title(
            "Silhouette width plot: n_clusters={nc}, silhouette_avg={s:.2f}".format(nc=n_clusters, s=silhouette_avg))
        ax_sh.set_xlabel("Silhouette score")
        ax_sh.set_ylabel("Cluster labels")
        #       cluster visualization
        ax_cluster.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='^', s=30, lw=0, alpha=0.7,
                           c=xcluster['cluster_label'].map(color_map), edgecolor='k')

        ax_cluster.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o',
                           c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(cluster_centers):
            ax_cluster.scatter(c[0], c[1], marker='${}$'.format(i), alpha=1,
                               s=50, edgecolor='k')
        ax_cluster.set_title(
            "clustered data using {m}: n_clusters={nc}".format(m=fit_res['cluster_method'], nc=n_clusters))
        ax_cluster.set_xlabel("1st feature space: {}".format(X.columns[0]))
        ax_cluster.set_ylabel("2nd feature space: {}".format(X.columns[1]))
        fig.tight_layout()
        return fig


