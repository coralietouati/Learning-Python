import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
sns.set(style="darkgrid")


def cluster( d, metric,feature,k,clustername):
    daily = d.groupby([feature, 'Hour']).mean().reset_index()
    matrix = daily.pivot_table(index=[feature],columns=['Hour'], values = [metric], fill_value=0)
    Z = linkage(matrix, 'ward')

    plt.figure(1)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    clusters = fcluster(Z, k, criterion='maxclust')
    features = matrix.index.get_values()
    dfclusters = pd.DataFrame({ feature: features, 'clusters': clusters} )
    # add cluster in d
    for feat in d[feature].unique():
        d.loc[d[feature] == feat, clustername] = dfclusters[dfclusters[feature] == feat].clusters.item()

    moyenne_c = d.groupby([feature,'Hour',clustername]).mean().reset_index()

    # Add color for culster
    color=['indianred','cornflowerblue','mediumpurple','orange']
    clus_nb = list(moyenne_c[clustername].unique())
    clus_nb.sort()
    for clus in clus_nb:
        dfclusters.loc[dfclusters.clusters == clus, 'color'] = color[int(clus)-1]

    plt.figure(2)
    sns.lineplot(x='Hour', y=metric, hue=clustername, data=moyenne_c,  palette=color[:k], alpha=0.4, legend=False).set_title(clustername)
    plt.figure(3)
    sns.lineplot(x='Hour', y=metric, hue=feature, data=moyenne_c,  palette=list(dfclusters['color']), alpha=0.3, legend=False).set_title(clustername)
    clustering = moyenne_c.groupby(['Hour',clustername]).mean().reset_index()
    sns.lineplot(x='Hour', y=metric, hue=clustername, palette=color[:k], data=clustering, linewidth=3)
    print(dfclusters)
    return dfclusters


def cluster_share(table):
    ratio = table.groupby('clusters').count().drop(['color'], axis=1)
    ratio['share'] = ratio.iloc[:,0]/ratio.iloc[:,0].sum()
    print(ratio)
    return ratio