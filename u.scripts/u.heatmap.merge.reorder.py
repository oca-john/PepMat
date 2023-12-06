import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# >>> 基于聚类合并相似的列，并绘制热图 //////////////////////////
qinhe_mt5 = pd.read_csv('a.qinhe_matrix_5.csv', index_col=0, header=0)
qinhe_clu = qinhe_mt5.values  # 获得数值区域，转为数组以便迭代计算


def merge_similar_columns(data, threshold):
    dist_matrix = pdist(data.T)  # 计算列之间的距离
    linkage_matrix = linkage(dist_matrix, method='ward')  # 基于ward算法聚类
    # 根据阈值合并聚类结果
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    # 合并相似性较高的列
    merged_data = []
    merged_columns = []
    for cluster_id in np.unique(clusters):
        cluster_columns = np.where(clusters == cluster_id)[0]  # 取相似的簇
        representative_column = cluster_columns[0]  # 以0号元素代替
        merged_data.append(data[:, representative_column])  # 从源数据中取出该列
        merged_columns.append(cluster_columns)  # 被合并的列，可选查看
    return np.column_stack(merged_data), merged_columns


data = qinhe_clu
threshold = 60

merged_data, merged_columns = merge_similar_columns(data, threshold)

# print("before_merged:")
# print(data)
# print("after_merged:")
# print(merged_data)

# 计算聚类并绘制热图（合并前后）
matplotlib.rc('font', family='Calibri', size=8)

# linked = linkage(data, 'ward')
# sns.clustermap(qinhe_mt5, row_linkage=linked, col_linkage=None,
#                cmap='Reds', cbar_pos=None, figsize=(3, 4))
sns.clustermap(qinhe_mt5, method='ward', metric='euclidean', cmap='Reds',
               row_cluster=True, col_cluster=True,  # cbar_pos=None,
               dendrogram_ratio=0.2, figsize=(4, 5))
# plt.savefig('a.qinhe_mt5_hm.jpg', dpi=300,
#             bbox_inches='tight', pad_inches=0)

# sns.clustermap(merged_data, row_linkage=linked, col_linkage=None,
#                cmap='Reds', cbar_pos=None, figsize=(3, 4))
heatmap = sns.clustermap(merged_data, method='ward', metric='euclidean',
                         cmap='Reds',
                         row_cluster=True, col_cluster=True,  # cbar_pos=None,
                         dendrogram_ratio=0.2, figsize=(3, 5))

# 获取横坐标轴和纵坐标轴的标签（已绘制-获取后重绘）
xtick_labels = heatmap.ax_heatmap.get_xticklabels()
ytick_labels = heatmap.ax_heatmap.get_yticklabels()
# print(type(xtick_labels), len(xtick_labels), "\n", xtick_labels)
# print(type(ytick_labels), len(ytick_labels), "\n", ytick_labels)

# 获取重新排序后的行列索引，根据索引获取对应的行列名（未绘制时，以此获得所有标签）
col_index = heatmap.dendrogram_col.reordered_ind
row_index = heatmap.dendrogram_row.reordered_ind
col_labels = qinhe_mt5.columns[col_index]  # 原始数据才有行列名
row_labels = qinhe_mt5.index[row_index]
# print(len(col_index), "\n", col_index)  # 合并后的数据包含500行18列（但不能全部展示）
# print(len(row_index), "\n", row_index)

# 根据行列索引从原始数据中获得合并且重排的数据
# reordered_data = qinhe_mt5.iloc[row_index, col_index]
# # print(type(reordered_data), reordered_data)
# reordered_data.to_csv('a.qinhe_mt5_mrg_ord.csv')

# 计算图中标签数和数据量的倍数，按标签数生成数列（作为索引）
col_bei = round(len(col_index) / len(xtick_labels))  # 18/18=1
x_index = list(range(0, len(xtick_labels)))
row_bei = round(len(row_index) / len(ytick_labels))  # 500/30=17
y_index = list(range(0, len(ytick_labels)))

# 按索引从原始标签中间隔性取出对应的标签，用于轴标签
# x_labels = [col_index[x*col_bei] for x in x_index]  # 用 col_index 检查索引正确
# y_labels = [row_index[x*row_bei] for x in y_index]
x_labels = [col_labels[x * col_bei] for x in x_index]  # 用 col_labels 更新标签
y_labels = [row_labels[x * row_bei] for x in y_index]
# print(x_labels)
# print(y_labels)

# 更新热图的行列名
heatmap.ax_heatmap.set_xticklabels(x_labels, rotation=90)
heatmap.ax_heatmap.set_yticklabels(y_labels)

# plt.savefig('a.qinhe_mt5_hm.merge.jpg', dpi=300, bbox_inches='tight',
#             pad_inches=0)
plt.show()
