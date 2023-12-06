import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


# 此程序用于将筛选获得的特异性亲和力矩阵按照指定区域和顺序填充到芯片上


# >>> 数据分组，计算倍率，整合为新数据框并可视化 /////////////////////////////////////
# 将读入的数据框转换为矩阵
affi = pd.read_csv("a.qinhe_norm_slec.csv", header=0, index_col=0)
# print(affi[:25])

# 每25行作为一个多肽组，用于一种气体的感知，放置在同一个区域
grp1 = affi[:25]
grp2 = affi[25:50]
grp3 = affi[50:75]
grp4 = affi[75:]
# print(grp1, grp2, grp3, grp4)
# print(grp1.iloc[:, 0].sum())

# 计算第一组第一列，第二组第二列，第三组第三列，第四组第四列的亲和力
inf_grp1 = grp1.iloc[:, 0].sum()
inf_grp2 = grp2.iloc[:, 1].sum()
inf_grp3 = grp3.iloc[:, 2].sum()
inf_grp4 = grp4.iloc[:, 3].sum()
# print(inf_grp1, inf_grp2, inf_grp3, inf_grp4)


# 将四个亲和力中最大的一个视为1,其他三个亲和力分别计算相对于最大值的倍数（三位小数）
max_affi = max(np.amax(inf_grp1), np.amax(inf_grp2),
               np.amax(inf_grp3), np.amax(inf_grp4))
affi1 = round(inf_grp1/max_affi, 3)
affi2 = round(inf_grp2/max_affi, 3)
affi3 = round(inf_grp3/max_affi, 3)
affi4 = round(inf_grp4/max_affi, 3)
# print(affi1, affi2, affi3, affi4)  # 1.0 0.868 0.917 0.844

# >>> 插队，基于此阶段的分组数据计算剩余12区块的相对亲和力 /////////////////////////////
# 计算其他12个分区亲和力相对于同一行中高亲和力区域的分数信息
# 即，用被标注区域除以最大亲和力区域，数值小于1，是个分数
# 后续可以基于芯片一个位置的信息求和，来推断此分区的多肽对其他气体的亲和力
# 即可对亮度进行修正（四者若都有反应则，需要适当削弱对应位置的亮度）
# 计算第一组第一列，第二组第二列，第三组第三列，第四组第四列的亲和力

# 注意此时所有数据均已被缩放为统一尺度，因此对角线数据也应使用缩放后的版本（大致都是1）

inf_grp11 = grp1.iloc[:, 1].sum()
inf_grp12 = grp1.iloc[:, 2].sum()
inf_grp13 = grp1.iloc[:, 3].sum()
affi11 = round(inf_grp11/max_affi, 3)
affi12 = round(inf_grp12/max_affi, 3)
affi13 = round(inf_grp13/max_affi, 3)
# print(affi11, affi12, affi13)  # 1.000  0.743  0.728  0.756

inf_grp20 = grp2.iloc[:, 0].sum()
inf_grp22 = grp2.iloc[:, 2].sum()
inf_grp23 = grp2.iloc[:, 3].sum()
affi20 = round(inf_grp20/max_affi, 3)
affi22 = round(inf_grp22/max_affi, 3)
affi23 = round(inf_grp23/max_affi, 3)
# print(affi20, affi22, affi23)  # 0.584  0.831  0.652  0.604

inf_grp30 = grp3.iloc[:, 0].sum()
inf_grp31 = grp3.iloc[:, 1].sum()
inf_grp33 = grp3.iloc[:, 3].sum()
affi30 = round(inf_grp30/max_affi, 3)
affi31 = round(inf_grp31/max_affi, 3)
affi33 = round(inf_grp33/max_affi, 3)
# print(affi30, affi31, affi33)  # 0.598  0.692  0.904  0.648

inf_grp40 = grp4.iloc[:, 0].sum()
inf_grp41 = grp4.iloc[:, 1].sum()
inf_grp42 = grp4.iloc[:, 2].sum()
affi40 = round(inf_grp40/max_affi, 3)
affi41 = round(inf_grp41/max_affi, 3)
affi42 = round(inf_grp42/max_affi, 3)
# print(affi40, affi41, affi42)  # 0.678  0.626  0.611  0.816

aff_tms_mtx = [[affi1, affi11, affi12, affi13],
               [affi20, affi2, affi22, affi23],
               [affi30, affi31, affi3, affi33],
               [affi40,  affi41, affi42, affi4]]
aff_tms_mtx = pd.DataFrame(aff_tms_mtx)
aff_tms_mtx.to_csv("a.qinhe_tms_mtx.csv", index=False, header=False)
print(aff_tms_mtx)
# 之前版本行内计算的结果，统一查看容易引起误解，以 (a1, b1) 重新计算
# >>> 插队结束 /////////////////////////////////////////////////////////////////


# 按照倍率缩放原始数据，使其具有一致的特异性亲和力区域
grp1 = grp1 / affi1
grp1 = grp1.where(grp1 <= 100, 100)
grp1 = grp1.round(2)
grp2 = grp2 / affi2
grp2 = grp2.where(grp2 <= 100, 100)
grp2 = grp2.round(2)
grp3 = grp3 / affi3
grp3 = grp3.where(grp3 <= 100, 100)
grp3 = grp3.round(2)
grp4 = grp4 / affi4
grp4 = grp4.where(grp4 <= 100, 100)
grp4 = grp4.round(2)
# 整合为一个表格，并将超过100的值替换为100
affin = pd.concat([grp1, grp2, grp3, grp4], ignore_index=False)
# # count = (affin > 100).sum().sum()
# # print(count)  # 有 3 个数值大于 100
# affin = affin.where(affin <= 100, 100)  # 改到前面分别修正各分组
# affin = affin.round(2)
# # count = (affin > 100).sum().sum()
# # print(count)  # 纠正后则不含大于 100 的值
affin.to_csv("a.qinhe_norm_suo.csv")
# print(affin)

# 用热图查看经过倍率平衡后的四个分组数据
plt.figure(figsize=(2, 2))
# matplotlib.rc('font', family='Calibri', size=8)
ax = sns.heatmap(affin, cmap='Reds', cbar=False)
ax.set_axis_off()
plt.xticks(rotation=20)
plt.savefig('a.qinhe_norm_suo_hm.jpg', dpi=300,
            bbox_inches='tight', pad_inches=0)
# plt.show()


# >>> 填充规则，包括亲和力序列和填充序列 ////////////////////////////////////////////
# 四个分组内部分别排序，获得芯片上四个区域内的多肽亲和力次序
grp1 = grp1.sort_values(by=grp1.columns[0], ascending=False)
grp2 = grp2.sort_values(by=grp2.columns[1], ascending=False)
grp3 = grp3.sort_values(by=grp3.columns[2], ascending=False)
grp4 = grp4.sort_values(by=grp4.columns[3], ascending=False)
affin2 = pd.concat([grp1, grp2, grp3, grp4], ignore_index=False)
affin2 = affin2.round(2)
affin2.to_csv("a.qinhe_norm_suo2.csv")

# 获取4个分组的索引并转为列表
ind1 = grp1.index.tolist()
ind2 = grp2.index.tolist()
ind3 = grp3.index.tolist()
ind4 = grp4.index.tolist()
# print(ind1, "\n", ind2, "\n", ind3, "\n", ind4)

# 用热图查看经过倍率平衡后的四个分组数据
plt.figure(figsize=(2, 2))
ax = sns.heatmap(affin2, cmap='Reds', cbar=False)
ax.set_axis_off()
plt.savefig('a.qinhe_norm_suo2_hm.jpg', dpi=300,
            bbox_inches='tight', pad_inches=0)
# plt.show()

# 设计四个分区内的填充次序，并将次序转为芯片坐标序列
# (X1, Y1)               # 子集1的坐标是从index=0开始计算的，比较容易
# (X2=Y1, Y2=9-X1)       # 9是基于矩阵边长计算的（边长10，最大index是9）
# (X3=9-X1, Y3=9-Y1)
# (X4=9-X1, Y4=Y1)
a1 = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
b1 = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 4, 3, 2, 4, 3, 4]
a2 = b1
b2 = [9 - x for x in a1]
a3 = [9 - x for x in a1]
b3 = [9 - y for y in b1]
a4 = [9 - x for x in a1]
b4 = b1
# print(a1, "\n", b1, "\n",  a2, "\n", b2, "\n", a3, "\n", b3,
#       "\n", a4, "\n", b4)


# >>> 执行填充多肽序列，获得芯片布局 ///////////////////////////////////////////////
# 将多肽依次填入芯片对应位置，获得芯片布局
mtx = [["" for _ in range(10)] for _ in range(10)]
for i in range(25):
    row = a1[i]
    col = b1[i]
    mtx[row][col] = ind1[i]
for i in range(25):
    row = a2[i]
    col = b2[i]
    mtx[row][col] = ind2[i]
for i in range(25):
    row = a3[i]
    col = b3[i]
    mtx[row][col] = ind3[i]
for i in range(25):
    row = a4[i]
    col = b4[i]
    mtx[row][col] = ind4[i]
mtx = pd.DataFrame(mtx)
# print(mtx)

# 绘制芯片上的多肽布局
fig, ax = plt.subplots(figsize=(5, 2))
matplotlib.rc('font', family='Calibri', size=6)
ax.set_xlim([0, 15])
ax.set_ylim([0, 10])
# 遍历字符串矩阵，并在每个格子中绘制对应的字符
for i in range(10):
    for j in range(10):
        rect = patches.Rectangle((j*1.5, i), 1.5, 1, linewidth=1,
                                 edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        if mtx[i][j] != '':
            ax.text(j*1.5 + 0.75, i + 0.5, mtx[i][j],  # 绘制字符
                    ha='center', va='center',  # rotation=0,
                    color='black')
ax.set_xticks([])
ax.set_yticks([])
# plt.savefig('b.mtx_layout_grid.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('b.mtx_layout_grid.jpg', dpi=300, bbox_inches='tight',
            pad_inches=0.02)
# plt.show()

# 保存芯片布局（即多肽名称矩阵）
mtx.to_csv('b.mtx_layout.csv', index=False,  header=False)


# >>> 参考亲和力矩阵 /////////////////////////////////////////////////////////////
# 将源数据亲和力填充到芯片上，形成参考亲和力矩阵
# 多肽矩阵某个元素与亲和力表中某个多肽一致时，使用该多肽的亲和力替换此多肽名
# 取出各分组中高特异性、高亲和力的列
haffi1 = grp1.iloc[:, 0]  # .round(0)
haffi2 = grp2.iloc[:, 1]  # .round(0)
haffi3 = grp3.iloc[:, 2]  # .round(0)
haffi4 = grp4.iloc[:, 3]  # .round(0)
# 同样按照填充列名的方式，将亲和力数据填充到矩阵中
mtx2 = [["" for _ in range(10)] for _ in range(10)]
for i in range(25):
    row = a1[i]
    col = b1[i]
    mtx2[row][col] = haffi1[i]
for i in range(25):
    row = a2[i]
    col = b2[i]
    mtx2[row][col] = haffi2[i]
for i in range(25):
    row = a3[i]
    col = b3[i]
    mtx2[row][col] = haffi3[i]
for i in range(25):
    row = a4[i]
    col = b4[i]
    mtx2[row][col] = haffi4[i]
mtx2 = pd.DataFrame(mtx2)
# print(mtx2)

# Plot, annot 显示数值注释，fmt 指定显示类型为 d 整数 .1f 一位浮点，
# cbar 定义 color bar 是否显示
plt.figure(figsize=(4, 4))
sns.heatmap(mtx2, annot=True, annot_kws={"size": 8},
            cmap="Reds", linewidths=0, cbar=False)
plt.xticks([])  # 手动定义轴标签为空
plt.yticks([])
plt.tight_layout()
plt.savefig("b.mtx2_ref_affi_hm.png", dpi=300,
            bbox_inches="tight", pad_inches=0)
# plt.show()

# 保存参考亲和力矩阵
mtx2.to_csv('b.mtx2_ref_affi.csv', index=False,  header=False)
