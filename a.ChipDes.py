# import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.cluster.hierarchy import linkage, fcluster
# from scipy.spatial.distance import pdist


# 此程序用于从原始的亲和力数据库中筛选对4种气体具有高特异性、高亲和力的多肽组合。
# 包括数据的预处理、基于亲和力谱的聚类和列（气体）合并、基于特异性考虑的行（多肽）过滤和选择。
# 同时输出了对应于文章各部分所需的热图。


# >>> 归一化至 0-100 区间 ///////////////////////////////////////////////////////
# # 将亲和力进行归一化（0-100），气体编号、肽名、亲和力数值
# qinhe = pd.read_csv("data_bak/500ORX505VOC_fill_zero.csv", header=None)
# # qinhe.shape  # (252500, 3)
# condition = qinhe.iloc[:, 2] == 0
# qinhe = qinhe.drop(qinhe[condition].index)  # 删除 0 项，后期数据没有 0，则不考虑
# # qinhe.shape  # (198549, 3)
#
# # 原始的极值计算
# va = qinhe.iloc[:, 2].max()  # 极大值为：-0.1
# # print(va)
# vb = qinhe.iloc[:, 2].min()  # 极小值为：-12.7
# # print(vb)
#
# # 计算检测限（归一化后的对应值） - num of limitation
# # 归一化之后，假设原始的亲和力边界值为 -5，新的边界值计算
# nlim = round(((-5) - (vb)) * 100 / ((va) - (vb)))  # 新的边界是 61
# # print(nlim)
#
# # 获取第三列数值，计算极差，归一化到 0-100 之间
# qinhe_trans = round((qinhe.iloc[:, 2] - qinhe.iloc[:, 2].min()) / (
#     qinhe.iloc[:, 2].max() - qinhe.iloc[:, 2].min()) * 100)
# # qh2.shape     # (198549,)
# # qh2.head(10)  # 查看部分数据
# # 数据取整
# qinhe_int = []
# for d in qinhe_trans:
#     qinhe_int.append(int(d))
# # 重新写入
# qinhe[2] = qinhe_int
# # print(qinhe.head(10))
# # qinhe.to_csv('a.qinhe100.csv', index=False, header=False)

# 最终的目标是使用原始数据筛选“四个气体”及“与之有亲和力的100个多肽”。
# 筛选第三列数值大于61的所有记录（行），按气体类别进行分组并统计记录数量（数值），
# 若记录数量超过100则进一步计算其数值分布的离散度（方差），筛选所有分组中方差较大的部分
# （亲和力分布较分散），统计剩下的合格的气体类别是否足够（超过4种），不足就降低标准重新处理，
# 够4种就整理表格导出文件。


# >>> 亲和力 5 个水平划分 ////////////////////////////////////////////////////////
# # 前述代码预处理数据并保存，后续直接从读取开始
# qinhe = pd.read_csv("data_bak/a.qinhe100.csv", header=None)
# # print(qinhe.head(10))       # a.qinhe100 就是归一化到 100 之后的版本
# # 后续直接使用计算过的新边界作为初个筛选条件，后续根据结果调整
# # nlim = 61  # 预设的阈值，筛选条件不能低于此阈值
#
# # 设计5个亲和力区间，并划分数据
# # 0-40, 41-55, 56-70, 71-85, 86-100
# for index, row in qinhe.iterrows():  # 比较耗时
#     if 86 <= row[2] <= 100:
#         qinhe.loc[index, 3] = 5
#     elif 71 <= row[2] <= 85:
#         qinhe.loc[index, 3] = 4
#     elif 56 <= row[2] <= 71:
#         qinhe.loc[index, 3] = 3
#     elif 41 <= row[2] <= 56:
#         qinhe.loc[index, 3] = 2
#     else:
#         qinhe.loc[index, 3] = 1
# # 保存按照5个水平分割后的数据
# # qinhe.to_csv('a.qinhe100div5.csv', index=False, header=False)


# >>> 亲和力数据框转矩阵（u.change.df.matrix.py） /////////////////////////////////
# # 亲和力数据（数据框）
# qinhe = pd.read_csv("data_bak/a.qinhe100div5.csv", header=None)
# data = qinhe.values  # 数据框转二维数组（否则for循环报错）
# # 获取气体和多肽的唯一值（气体和多肽列表）
# gases = list(set([row[0] for row in data]))
# peptides = list(set([row[1] for row in data]))
#
# # 创建空矩阵
# matrix = np.zeros((len(peptides), len(gases)))
# # 填充矩阵
# for row in data:
#     gas = row[0]
#     peptide = row[1]
#     affinity = row[3]   # 第4列是5分类数值
#     # affinity = row[2]  # 第3列是归一化数值
#     # 将原表第一二列转为矩阵索引
#     gas_index = gases.index(gas)
#     peptide_index = peptides.index(peptide)
#     # 按照索引位置取亲和力
#     matrix[peptide_index, gas_index] = affinity
#
# # 创建带有行名和列名的 DataFrame
# matrix_df = pd.DataFrame(matrix, index=peptides, columns=gases)
#
# # 输出转换后的矩阵
# # matrix_df.to_csv("a.qinhe_matrix_5.csv")
# # matrix_df.to_csv("a.qinhe_matrix_100.csv")
# # print(matrix_df)


# >>> 绘制亲和力聚类热图 /////////////////////////////////////////////////////////
# # 读入亲和力矩阵
# qinhe_mt5 = pd.read_csv('a.qinhe_matrix_5.csv', index_col=0, header=0)
# qinhe_mt5_clu = qinhe_mt5.values
# # 根据矩阵计算聚类信息
# linked = linkage(qinhe_mt5_clu, 'ward')
#
# # seaborn 绘制
# matplotlib.rc('font', family='Calibri', size=8)
# sns.clustermap(qinhe_mt5, row_linkage=linked, col_linkage=None,
#                cmap='Reds', cbar_pos=None, figsize=(5.5, 5.5))
# # 5.5 英寸大约是 14 厘米是 BioSensors 的行宽
# # plt.savefig("a.qinhe_mt5_cluster.pdf", bbox_inches='tight', pad_inches=0)
# # plt.savefig("a.qinhe_mt5_cluster.jpg", dpi=300, bbox_inches="tight",
# #             pad_inches=0)
# plt.show()


# >>> 基于聚类合并相似的列，并绘制热图 //////////////////////////////////////////////
# qinhe_mt5 = pd.read_csv('a.qinhe_matrix_5.csv', index_col=0, header=0)
# qinhe_clu = qinhe_mt5.values  # 获得数值区域，转为数组以便迭代计算
#
#
# def merge_similar_columns(data, threshold):
#     dist_matrix = pdist(data.T)  # 计算列之间的距离
#     linkage_matrix = linkage(dist_matrix, method='ward')  # 基于ward算法聚类
#     # 根据阈值合并聚类结果
#     clusters = fcluster(linkage_matrix, threshold, criterion='distance')
#     # 合并相似性较高的列
#     merged_data = []
#     merged_columns = []
#     for cluster_id in np.unique(clusters):
#         cluster_columns = np.where(clusters == cluster_id)[0]  # 取相似的簇
#         representative_column = cluster_columns[0]  # 以0号元素代替
#         merged_data.append(data[:, representative_column])  # 从源数据中取出该列
#         merged_columns.append(cluster_columns)  # 被合并的列，可选查看
#     return np.column_stack(merged_data), merged_columns
#
#
# data = qinhe_clu
# threshold = 55  # 【列合并阈值 —— 第一个重要筛选参数】===============================
#
# merged_data, merged_columns = merge_similar_columns(data, threshold)
#
# # print("before_merged:")
# # print(data)
# # print("after_merged:")
# # print(merged_data)
#
# # # 计算聚类并绘制热图（合并前后）
# # matplotlib.rc('font', family='Calibri', size=8)
# #
# # # linked = linkage(data, 'ward')
# # # sns.clustermap(qinhe_mt5, row_linkage=linked, col_linkage=None,
# # #                cmap='Reds', cbar_pos=None, figsize=(3, 4))
# # sns.clustermap(qinhe_mt5, method='ward', metric='euclidean', cmap='Reds',
# #                row_cluster=True, col_cluster=True,  # cbar_pos=None,
# #                dendrogram_ratio=0.2, figsize=(4, 5))
# # plt.savefig('a.qinhe_mt5_hm.jpg', dpi=300,
# #             bbox_inches='tight', pad_inches=0)
#
# # sns.clustermap(merged_data, row_linkage=linked, col_linkage=None,
# #                cmap='Reds', cbar_pos=None, figsize=(3, 4))
# heatmap = sns.clustermap(merged_data, method='ward', metric='euclidean',
#                          cmap='Reds', row_cluster=True, col_cluster=True,
#                          # cbar_pos=None,
#                          dendrogram_ratio=0.2, figsize=(3, 5))
#
# # 获取横坐标轴和纵坐标轴的标签（已绘制-获取后重绘）
# xtick_labels = heatmap.ax_heatmap.get_xticklabels()
# ytick_labels = heatmap.ax_heatmap.get_yticklabels()
# # print(type(xtick_labels), len(xtick_labels), "\n", xtick_labels)
# # print(type(ytick_labels), len(ytick_labels), "\n", ytick_labels)
#
# # 获取重新排序后的行列索引，根据索引获取对应的行列名（未绘制时，以此获得所有标签）
# col_index = heatmap.dendrogram_col.reordered_ind
# row_index = heatmap.dendrogram_row.reordered_ind
# col_labels = qinhe_mt5.columns[col_index]  # 原始数据才有行列名
# row_labels = qinhe_mt5.index[row_index]
# # print(len(col_index), "\n", col_index)  # 合并后的数据包含500行18列（但不能全部展示）
# # print(len(row_index), "\n", row_index)
#
# # 根据行列索引从原始数据中获得合并且重排的数据
# reordered_data = qinhe_mt5.iloc[row_index, col_index]
# # print(type(reordered_data), reordered_data)
# reordered_data.to_csv('a.qinhe_mt5_mrg.csv')
#
# # # 计算图中标签数和数据量的倍数，按标签数生成数列（作为索引）
# # col_bei = round(len(col_index)/len(xtick_labels))  # 18/18=1
# # x_index = list(range(0, len(xtick_labels)))
# # row_bei = round(len(row_index)/len(ytick_labels))  # 500/30=17
# # y_index = list(range(0, len(ytick_labels)))
# #
# # # 按索引从原始标签中间隔性取出对应的标签，用于轴标签
# # # x_labels = [col_index[x*col_bei] for x in x_index]  # 用 col_index 检查索引正确
# # # y_labels = [row_index[x*row_bei] for x in y_index]
# # x_labels = [col_labels[x*col_bei] for x in x_index]  # 用 col_labels 更新标签
# # y_labels = [row_labels[x*row_bei] for x in y_index]
# # # print(x_labels)
# # # print(y_labels)
# #
# # # 更新热图的行列名
# # heatmap.ax_heatmap.set_xticklabels(x_labels, rotation=90)
# # heatmap.ax_heatmap.set_yticklabels(y_labels)
# # # plt.savefig('a.qinhe_mt5_mrg_hm.jpg', dpi=300, bbox_inches='tight',
# # #             pad_inches=0)
# # plt.show()


# >>> 亲和力数据框转矩阵（u.change.df.matrix.py） /////////////////////////////////
# 读入或继承前述筛选的数据表
qinhe_norm = pd.read_csv('a.qinhe_matrix_100.csv', index_col=0, header=0)
ord_data = pd.read_csv('a.qinhe_mt5_mrg.csv', index_col=0, header=0)
# print(ord_data.head(5))  # 导入的表格包含64列

# 删除众数率超过 0.5 的列【众数阈值 —— 第二个重要筛选参数】============================
m_counts = ord_data.apply(lambda x: x.value_counts().max(), axis=0)
n1_data = ord_data.drop(m_counts[m_counts > len(ord_data) * 0.5].index, axis=1)
# print(n1_data.head(5))  # 删除后37列

# 初始化结果字典
result = {}

# 获取所有列的组合
columns = n1_data.columns.tolist()
combinations = list(combinations(columns, 4))
# 按 0.5 筛选后组合获得了 1820 种潜在方案
# 按 0.6 筛选后组合获得了 14950 种潜在方案
# 重新按50相似度合并相似的列，按 0.5 众数率筛选后组合获得了 66045 种潜在方案
# print(len(combinations))

# 若不命名组合，则后续必须知道组合的内容，才能按照该键取出相应的条目
# 设置编号，则可以根据向字典中添加的条目数对字典内容进行编号，可方便取出
update_count = 1

# 处理每个组合
for combo in combinations:
    # 获取组合对应的列名
    col_names = list(combo)
    # print(type(col_names), col_names)  # list 类型，第一种组合包含的4个列名
    # 先将4列数据取出来再操作，否则直接操作 ord_data 是 32 列的源数据（无法调和将会是所有数据被删）
    for_data = ord_data[col_names]
    # 取出相应的列并删除所有包含空值的行
    slec = for_data[(for_data != 0).all(1)]  # 删除包含0的行，已验证
    # print(type(slec), slec.shape)  # dataframe 类型，第一种组合包含的4列数据表
    # 删除空值行后(436,4)

    # 根据条件生成8个集合

    # 方案一：高亲和列为4,5，其他列为1,2,3
    # 按照高亲和力和低亲和力出现的列逐个集合筛选
    # 众数删除阈值设置0.6，获得组合14950，最终筛出1种可行方案（包含6种高亲和力多肽）
    # 在数据处理时只能对边角边长为3的格子卷积，其他位置的卷积反而会降低指示性
    # set1 = slec.index[(slec[col_names[0]].isin([4, 5]))
    #             & ((slec[col_names[1]].isin([1, 2, 3]))
    #             & (slec[col_names[2]].isin([1, 2, 3]))
    #             & (slec[col_names[3]].isin([1, 2, 3])))]
    # set2 = slec.index[(slec[col_names[0]].isin([1, 2, 3]))
    #             & ((slec[col_names[1]].isin([4, 5]))
    #             & (slec[col_names[2]].isin([1, 2, 3]))
    #             & (slec[col_names[3]].isin([1, 2, 3])))]
    # set3 = slec.index[(slec[col_names[0]].isin([1, 2, 3]))
    #             & ((slec[col_names[1]].isin([1, 2, 3]))
    #             & (slec[col_names[2]].isin([4, 5]))
    #             & (slec[col_names[3]].isin([1, 2, 3])))]
    # set4 = slec.index[(slec[col_names[0]].isin([1, 2, 3]))
    #             & ((slec[col_names[1]].isin([1, 2, 3]))
    #             & (slec[col_names[2]].isin([1, 2, 3]))
    #             & (slec[col_names[3]].isin([4, 5])))]
    # # print(set1, "\n", set2, "\n", set3, "\n", set4)
    # # break
    # 45的"列相似性阈值"和0.5的"众数重复率阈值"获得了292825种组合，但仍未筛选出符合条件的方案
    # 推测此方案过于严格，暂使用方案二

    # 方案二：
    # 获取各列中为该行最大值（即采用相对大小）所在的行号列表，并保存行名
    set1 = slec.index[slec[col_names[0]] == slec.max(axis=1)]
    set2 = slec.index[slec[col_names[1]] == slec.max(axis=1)]
    set3 = slec.index[slec[col_names[2]] == slec.max(axis=1)]
    set4 = slec.index[slec[col_names[3]] == slec.max(axis=1)]
    # print(set1, "\n", set2, "\n", set3, "\n", set4)
    # 55,0.5的参数获得了12650种组合，获得70种方案

    # 计算每个集合独有元素
    # set1 = set1 差 (set2 并 set3 并 set4)
    set1_ind = set1.difference(set2.union(set3.union(set4)))
    set2_ind = set2.difference(set1.union(set3.union(set4)))
    set3_ind = set3.difference(set2.union(set1.union(set4)))
    set4_ind = set4.difference(set2.union(set3.union(set1)))
    # 整合所有结果为一个二维数组
    val_set = [col_names, set1_ind, set2_ind, set3_ind, set4_ind]
    # print(set1_ind, "\n", set2_ind, "\n", set3_ind, "\n", set4_ind)
    # print(len(set1_ind), len(set2_ind), len(set3_ind), len(set4_ind))
    # print(val_set)

    # 若每个独有元素都超过25个则认为可以满足区分亲和力的要求
    if (len(set1_ind) >= 25 and len(set2_ind) >= 25 and len(set3_ind) >= 25 and
            len(set4_ind) >= 25):
        result[update_count] = val_set  # 更新整合的方案
        update_count += 1  # 计数器作为方案编号
    # break

# 输出结果字典
# 列合并阈值60，众数率阈值0.5，共1820组合
# 方法一，获得 0 个方案
# 方法二，获得了 1 个可用的方案(gas4最多获得21种多肽，需要重用部分多肽补全)
res_df = pd.DataFrame(result).T
# res_df.to_csv('a.sche_result.csv', index=True, header=False)
# print(len(result))


# >>> 基于亲和力筛选结果的 /////////////////////////////////
# 把从字典中抽取方案并进行芯片设计的部分单独运行，避免每次计算消耗过多时间
# 此部分可以注释之前的代码运行，或以独立文件的形式运行

# 读取需要先读出来，转置后转为字典格式
# res = pd.read_csv('a.sche_result.csv', index_col=0)
# result = res.T.to_dict()  # 只能解析一层字典，不能解析嵌套结构

# 目前共70种备选方案，利用方案编号快速选择数据，以便查看各列之间的差异性
sche_nu = 8  # 1-70范围内的任意整数编号
# 方案 4, 6, 7 表现较差
# 方案 1, 8 表现较好

gases = result[sche_nu][0]
pep_gas1 = list(result[sche_nu][1])
gas1_pep = pep_gas1[0:25]
pep_gas2 = list(result[sche_nu][2])
gas2_pep = pep_gas2[0:25]
pep_gas3 = list(result[sche_nu][3])
gas3_pep = pep_gas3[0:25]
pep_gas4 = list(result[sche_nu][4])
gas4_pep = pep_gas4[0:25]
# 检查数据类型（统一转为列表）
# print(type(gases), "\n", type(pep_gas1), "\n", type(pep_gas2), "\n",
#       type(pep_gas3), "\n", type(pep_gas4))
# print(gases, "\n", pep_gas1, "\n", pep_gas2, "\n", pep_gas3, "\n", pep_gas4)
# 检查4个气体分别有多少个特异性气体
# print(len(pep_gas1), len(pep_gas2), len(pep_gas3), len(pep_gas4))
# print(gases, "\n", gas1_pep, "\n", gas2_pep, "\n", gas3_pep, "\n", gas4_pep)

# 根据上述结果构造新的表格
# 合并行名信息，使用4个25的筛选方案，则不需要复用数据
peps = gas1_pep + gas2_pep + gas3_pep + gas4_pep  # + gas4_pep[:4]


# 构造新矩阵

# 输出 5 水平版本的矩阵，并以热图和数值矩阵检查
new_data = ord_data.loc[peps, gases]
new_data.to_csv("a.qinhe_mt5_slec.csv")
# 用热图查看最后的结果是否有区分度
plt.figure(figsize=(2, 2))
matplotlib.rc('font', family='Calibri', size=8)  # heatmap 函数需要在内部设置
ax = sns.heatmap(new_data, cmap='Reds', cbar=False)
ax.set_axis_off()
# plt.xticks(rotation=20)
plt.savefig('a.qinhe_mt5_slec_hm.jpg', dpi=300,
            bbox_inches='tight', pad_inches=0)
# plt.show()
# print(type(new_data), "\n", new_data)

# # 输出归一化版本的矩阵，并以热图和数值矩阵检查
new_data2 = qinhe_norm.loc[peps, gases]
new_data2.to_csv("a.qinhe_norm_slec.csv")
# 用热图查看最后的结果是否有区分度
plt.figure(figsize=(2, 2))
# matplotlib.rc('font', family='Calibri', size=8)
ax2 = sns.heatmap(new_data2, cmap='Reds', cbar=False)
ax2.set_axis_off()
# plt.xticks(rotation=20)
plt.savefig('a.qinhe_norm_slec_hm.jpg', dpi=300,
            bbox_inches='tight', pad_inches=0)
# plt.show()
# print(type(new_data), "\n", new_data)
