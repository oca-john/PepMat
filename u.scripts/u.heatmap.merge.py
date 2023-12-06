import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# 用于聚类热图相同的算法聚类，并合并相似的列（减少重复性的数据）
# 绘制聚类前后的热图，展示我们合并后依然有很多可用的列（足够多有差异的模式）
# 具体使用时，由于选择代表性的列具有随机性（本身为了减少计算量，因此无法评估）

# 处理数据
data = np.random.rand(8, 32)  # 创建一个8行32列的随机数据矩阵
data = np.round(data*100).astype(int)


def merge_similar_columns(data, threshold):
    # 使用Ward算法进行列聚类
    dist_matrix = pdist(data.T)  # 计算列之间的距离
    linkage_matrix = linkage(dist_matrix, method='ward')  # 进行聚类
    # 根据指定的阈值合并聚类结果
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    # 合并相似性较高的列，使用任意一列作为代表
    merged_data = []
    merged_columns = []
    for cluster_id in np.unique(clusters):
        cluster_columns = np.where(clusters == cluster_id)[0]
        representative_column = cluster_columns[0]
        merged_data.append(data[:, representative_column])
        merged_columns.append(cluster_columns)
    return np.column_stack(merged_data), merged_columns


threshold = 95

merged_data, merged_columns = merge_similar_columns(data, threshold)

# print("before_merged:")
# print(data)
# print("after_merged:")
# print(merged_data)
# print("merged_cols:")
# print(merged_columns)

# 无聚类热图
plt.figure(figsize=(3, 3))
matplotlib.rc('font', family='Calibri', size=8)
ax = sns.heatmap(data, cmap='Greys', cbar=False)
ax.set_axis_off()
plt.savefig('heatmap.dat0.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

# 计算聚类并绘制热图（合并前后）
linked = linkage(data, 'ward')
# matplotlib.rc('font', family='Calibri', size=8)
sns.clustermap(data, row_linkage=linked, col_linkage=None, xticklabels=False,
               yticklabels=False, cmap='Greys', cbar_pos=None,
               dendrogram_ratio=0.1, figsize=(3, 3))
plt.savefig('heatmap.data.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

# matplotlib.rc('font', family='Calibri', size=8)
# linked2 = linkage(merged_data, 'ward')
sns.clustermap(merged_data, row_linkage=linked, col_linkage=None,
               xticklabels=False, yticklabels=False, cmap='Greys',
               cbar_pos=None, dendrogram_ratio=0.1, figsize=(3, 3))
plt.savefig('heatmap.data.merge.jpg', dpi=300, bbox_inches='tight',
            pad_inches=0)
plt.show()
