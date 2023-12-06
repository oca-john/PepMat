import numpy as np
import pandas as pd

# 将三列亲和力数据表，转换为矩阵形式
# 理解更直观，也可以转换为数据框，方便用 pandas 操作

# 亲和力数据（数据框）
df = pd.read_csv("a.qinhe100div5.csv", header=None)
data = df.values  # 数据框转二维数组（否则for循环报错）
# 获取气体和多肽的唯一值（气体和多肽列表）
gases = list(set([row[0] for row in data]))
peptides = list(set([row[1] for row in data]))

# 创建空矩阵
matrix = np.zeros((len(peptides), len(gases)))
# 填充矩阵
for row in data:
    gas = row[0]
    peptide = row[1]
    affinity = row[3]   # 第4列是5分类数值
    # affinity = row[2]  # 第3列是归一化数值
    # 将原表第一二列转为矩阵索引
    gas_index = gases.index(gas)
    peptide_index = peptides.index(peptide)
    # 按照索引位置取亲和力
    matrix[peptide_index, gas_index] = affinity

# 创建带有行名和列名的 DataFrame
matrix_df = pd.DataFrame(matrix, index=peptides, columns=gases)

# 输出转换后的矩阵
matrix_df.to_csv("a.qinhe_matrix_5.csv")
# matrix_df.to_csv("a.qinhe_matrix_100.csv")
# print(matrix_df)
