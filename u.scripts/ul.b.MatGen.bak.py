# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===== Part 1 亲和力列表提取 ===============================
# 自定义的亲和力列表（可从真实数据中归纳）
affi = pd.read_csv("ul.b.pep_qinhe.csv", header=0)
# print(affi.head(10))
# print(affi.columns)  # 获得列名信息

# 读取 pep 名列表，转为 pep 集合
pep = set(affi["pep_name"])
# print(type(pep))  # <class 'set'>
# print(pep)
# pep 信息保存为文件
peplist = pd.DataFrame(pep)
peplist = peplist[0]
# print(type(peplist))  # <class 'set'>
# print(peplist.shape)
peplist.to_csv("b.pepname.csv", index=False, header=None)

# 读取列名，后四列转为 gas 集合
gas = set(affi.columns[-4:])
# print(type(gas))    # <class 'set'>
# print(gas)          # {'affinity_with_gas1', 'affinity_with_gas3',
#                       'affinity_with_gas2', 'affinity_with_gas4'}
# gas 信息保存为文件
gaslist = pd.DataFrame(gas)
gaslist = gaslist[0]
# print(type(gaslist))  # <class 'set'>
# print(gaslist.shape)
gaslist.to_csv("b.gasname.csv", index=False, header=None)


# ===== Part 2 映射规则定义 =================================
# 从上表中筛选每个亲和力关系大于60的多肽作为具备亲和力的对象
g1p_tb = (
    affi[affi["affinity_gas1"] > 60]
    .filter(items=["pep_name", "affinity_gas1"])
    .sort_values(by="affinity_gas1", ascending=False)
)
# print(g1p_tb)
# print(g1p_tb.shape) # 获得了 28 个亲和力在 75 以上的
# 获得了 53 个亲和力在 60 以上的

g2p_tb = (
    affi[affi["affinity_gas2"] > 60]
    .filter(items=["pep_name", "affinity_gas2"])
    .sort_values(by="affinity_gas2", ascending=False)
)
# print(g2p_tb)
# print(g2p_tb.shape) # 获得了 22 个亲和力在 75 以上的
# 获得了 36 个亲和力在 60 以上的

g3p_tb = (
    affi[affi["affinity_gas3"] > 60]
    .filter(items=["pep_name", "affinity_gas3"])
    .sort_values(by="affinity_gas3", ascending=False)
)
# print(g3p_tb)
# print(g3p_tb.shape) # 获得了 20 个亲和力在 75 以上的
# 获得了 38 个亲和力在 60 以上的

g4p_tb = (
    affi[affi["affinity_gas4"] > 60]
    .filter(items=["pep_name", "affinity_gas4"])
    .sort_values(by="affinity_gas4", ascending=False)
)
# print(g4p_tb)
# print(g4p_tb.shape) # 获得了 30 个亲和力在 75 以上的
# 获得了 42 个亲和力在 60 以上的


# 从排序后的列表中提取亲和力信息，构建总集
g1a = g1p_tb["pep_name"]  # 亲和力，已验证了排序结果，g1a 指示临时变量，无意义
g1p = set(g1a)  # 集合，g1p 表示 gas1 和所有 pep 的亲和力（父集）
# print(len(g1p))       # 集合长度为 28

g2a = g2p_tb["pep_name"]  # 亲和力
g2p = set(g2p_tb["pep_name"])  # 集合
# print(len(g2p))     # 集合长度为 22

g3a = g3p_tb["pep_name"]  # 亲和力
g3p = set(g3p_tb["pep_name"])  # 集合
# print(len(g3p))     # 集合长度为 20

g4a = g4p_tb["pep_name"]  # 亲和力
g4p = set(g4p_tb["pep_name"])  # 集合
# print(len(g4p))     # 集合长/度为 30


# 子集计算（独享元素占据芯片角落，2类共享占据边缘，多类共享放中间）
# 单集合独享元素
s1 = g1p - (g2p | g3p | g4p)  # 左上，s1 表示 subset，是区分了其他集合重复元素的独享元素
s2 = g2p - (g1p | g3p | g4p)  # 右上
s3 = g3p - (g2p | g1p | g4p)  # 左下
s4 = g4p - (g2p | g3p | g1p)  # 右下

# 两集合共享元素
s12 = (g1p | g2p) - (g3p | g4p) - s1 - s2  # 两两之间的元素集合
s13 = (g1p | g3p) - (g2p | g4p) - s1 - s3
s24 = (g2p | g4p) - (g1p | g3p) - s2 - s4
s34 = (g3p | g4p) - (g1p | g2p) - s3 - s4

# 其他元素
sctr = (g1p | g2p | g3p | g4p) - (s1 | s2 | s3 | s4 | s12 | s13 | s24 | s34)

# print(len(s1), len(s2), len(s3), len(s4))       # 15 10 5 11
# print(len(s12), len(s13), len(s24), len(s34))   # 1 2 6 2
# print(len(sctr))                # 14
# 41 + 11 + 14，共 66 个与他们亲和力在 75 以上，剩余 34 个肽没有被收录在任何亲和力集合中

# print(len(s1), len(s2), len(s3), len(s4))       # 15 7 5 7
# print(len(s12), len(s13), len(s24), len(s34))   # 4 9 7 4
# print(len(sctr))                # 32
# 34 + 24 + 32，共 90 个与他们亲和力在 60 以上，剩余 10 个肽没有被收录在任何亲和力集合中


# 子集导出（用于数据生成的部分）
# subdir1 = "sets"
# os.makedirs(subdir1, exist_ok=True)
#
# s1_ls = pd.DataFrame(s1)
# filename = os.path.join(subdir1, 'c.s1.csv')
# s1_ls.to_csv(filename, index=False, header=None)
# s2_ls = pd.DataFrame(s2)
# filename = os.path.join(subdir1, 'c.s2.csv')
# s2_ls.to_csv(filename, index=False, header=None)
# s3_ls = pd.DataFrame(s3)
# filename = os.path.join(subdir1, 'c.s3.csv')
# s3_ls.to_csv(filename, index=False, header=None)
# s4_ls = pd.DataFrame(s4)
# filename = os.path.join(subdir1, 'c.s4.csv')
# s4_ls.to_csv(filename, index=False, header=None)
#
# s12_ls = pd.DataFrame(s12)
# filename = os.path.join(subdir1, 'c.s12.csv')
# s12_ls.to_csv(filename, index=False, header=None)
# s13_ls = pd.DataFrame(s13)
# filename = os.path.join(subdir1, 'c.s13.csv')
# s13_ls.to_csv(filename, index=False, header=None)
# s24_ls = pd.DataFrame(s24)
# filename = os.path.join(subdir1, 'c.s24.csv')
# s24_ls.to_csv(filename, index=False, header=None)
# s34_ls = pd.DataFrame(s34)
# filename = os.path.join(subdir1, 'c.s34.csv')
# s34_ls.to_csv(filename, index=False, header=None)
#
# sctr_ls = pd.DataFrame(sctr)
# filename = os.path.join(subdir1, 'c.sctr.csv')
# sctr_ls.to_csv(filename, index=False, header=None)


# ===== Part 3 肽探针布局 ==================================
# 定义一个 10*10 的初始化空数据框（每项填充控制 Na），即空板
pep_loc = pd.DataFrame(
    # np.empty(100,object) 生成空值列表
    np.zeros(100, object).reshape(10, 10)
)
# print(pep_loc.shape)
# print(pep_loc)

# 自动化实现部分，可能考虑根据实际的芯片位点规模自动设置空板，并从四个角位开始生成下列坐标数据
# 之后顺便计算所有数据的形状或长度，用于后续确定相关参数的上限

# 另一个思路——之前设想的坐标系旋转思路。只设计一个坐标系统，通过围绕中心点的旋转获得其他坐标系统。
# （已实现）具体实现方式是归纳四个坐标集合的复用思路。
# (X1, Y1)               # 子集1的坐标是从index=0开始计算的，比较容易
# (X2=Y1, Y2=9-X1)       # 9是基于矩阵边长计算的（边长10，最大index是9）
# (X3=9-X1, Y3=9-Y1)
# (X4=9-X1, Y4=Y1)

# 取四分之一大小的方格，从敏感角开始填充数据，定义带顺序的 5*5 位置信息列表
a1all = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 4, 3, 2, 4, 3,
         4]
b1all = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4,
         4]
a2all = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4,
         4]
b2all = [9, 8, 9, 7, 8, 9, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 5, 6, 7, 5, 6,
         5]
a3all = [9, 8, 9, 7, 8, 9, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 5, 6, 7, 5, 6,
         5]
b3all = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4,
         4]
a4all = [9, 8, 9, 7, 8, 9, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 5, 6, 7, 5, 6,
         5]
b4all = [9, 9, 8, 9, 8, 7, 9, 8, 7, 6, 9, 8, 7, 6, 5, 8, 7, 6, 5, 7, 6, 5, 6, 5,
         5]

# 填充 s1 区域，即气体 1 亲和力最强的列表
# g1a_s = np.array(g1a).remove(s1)  # 转 array
# g1a_s = list(g1a).remove(s1)      # 转 list
# 将 g1a 转为集合，获得差集，转为列表，从 g1a 中删除差集的元素
chaji1 = set(g1a) - set(s1)  # 获得并验证差集
# print(len(g1a),len(s1),len(chaji1))

g1a_s = list(g1a)
for i in chaji1:  # 最终的独享元素，带排序，15
    g1a_s.remove(i)
# print(g1a_s)    # ['pep95', 'pep14', 'pep25', 'pep38', 'pep35', 'pep76',
# 'pep67', 'pep93', 'pep17', 'pep50', 'pep33',
# 'pep36', 'pep2', 'pep23', 'pep98']
# print(g1a)      # 53 元素的原始 list

# 按 g1a_s 独享集合的尺寸，从两个坐标中分别取坐标，从肽列表取肽的别名
a1 = a1all[0: len(g1a_s)]
b1 = b1all[0: len(g1a_s)]
for a, b, nm in zip(a1, b1, g1a_s):
    pep_loc.iloc[a, b] = nm
# print(pep_loc)

# 以同样方式填写其他三个模块
chaji2 = set(g2a) - set(s2)
g2a_s = list(g2a)
for i in chaji2:  # 最终的独享元素，带排序，7
    g2a_s.remove(i)
a2 = a2all[0: len(g2a_s)]
b2 = b2all[0: len(g2a_s)]
for a, b, nm in zip(a2, b2, g2a_s):
    pep_loc.iloc[a, b] = nm

chaji3 = set(g3a) - set(s3)
g3a_s = list(g3a)
for i in chaji3:  # 最终的独享元素，带排序，5
    g3a_s.remove(i)
a3 = a3all[0: len(g3a_s)]
b3 = b3all[0: len(g3a_s)]
for a, b, nm in zip(a3, b3, g3a_s):
    pep_loc.iloc[a, b] = nm

chaji4 = set(g4a) - set(s4)
g4a_s = list(g4a)
for i in chaji4:  # 最终的独享元素，带排序，7
    g4a_s.remove(i)
a4 = a4all[0: len(g4a_s)]
b4 = b4all[0: len(g4a_s)]
for a, b, nm in zip(a4, b4, g4a_s):
    pep_loc.iloc[a, b] = nm
# print(pep_loc)

# 绘制格子化的布局图像（非矩阵）
# # 创建一个10x10的白色背景图像
# canvas = np.ones((10, 10, 3))
# # 绘制格子图像
# plt.figure(figsize=(3, 3))
# plt.axis('off')
# plt.imshow(canvas)
# # 绘制黑色格子线条
# for i in range(11):
#     plt.plot([-0.5, 9.5], [i - 0.5, i - 0.5], color='black', linewidth=1)
#     plt.plot([i - 0.5, i - 0.5], [-0.5, 9.5], color='black', linewidth=1)
# # 将字符串填入格子中
# for i in range(10):
#     for j in range(10):
#         text = pep_loc[i][j]
#         plt.text(j, i, text, fontsize=5, va='center', ha='center')
# # plt.show()
# plt.savefig("b.pep_loc0.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("b.pep_loc0.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02)


# 二集合交集，跟在独享集合之后继续填充
# 根据二集合填充起点，选择临近的方格继续填充（取索引时，需要排除前面已占用的部分）
# s13 -> s1
chaji12 = set(g1a) - set(s13)  # 获得并验证差集
g1a_s2 = list(g1a)
for i in chaji12:  # 最终的 s13 共享元素，带排序，9
    g1a_s2.remove(i)
# print(g1a_s2)   # ['pep57', 'pep87', 'pep19', 'pep32', 'pep51', 'pep89',
# 'pep53', 'pep28', 'pep47']
a12 = a1all[len(g1a_s): (len(g1a_s) + len(g1a_s2))]
b12 = b1all[len(g1a_s): (len(g1a_s) + len(g1a_s2))]
for a, b, nm in zip(a12, b12, g1a_s2):
    pep_loc.iloc[a, b] = nm
# print(pep_loc)

# s12 -> s2
chaji22 = set(g2a) - set(s12)
g2a_s2 = list(g2a)
for i in chaji22:  # 最终的 s12 共享元素，带排序，4
    g2a_s2.remove(i)
a22 = a2all[len(g2a_s): (len(g2a_s) + len(g2a_s2))]
b22 = b2all[len(g2a_s): (len(g2a_s) + len(g2a_s2))]
for a, b, nm in zip(a22, b22, g2a_s2):
    pep_loc.iloc[a, b] = nm

# s34 -> s3
chaji32 = set(g3a) - set(s34)
g3a_s2 = list(g3a)
for i in chaji32:  # 最终的 s34 共享元素，带排序，4
    g3a_s2.remove(i)
a32 = a3all[len(g3a_s): (len(g3a_s) + len(g3a_s2))]
b32 = b3all[len(g3a_s): (len(g3a_s) + len(g3a_s2))]
for a, b, nm in zip(a32, b32, g3a_s2):
    pep_loc.iloc[a, b] = nm

# s24 -> s4
chaji42 = set(g4a) - set(s24)
g4a_s2 = list(g4a)
for i in chaji42:  # 最终的 s24 共享元素，带排序，7
    g4a_s2.remove(i)
a42 = a4all[len(g4a_s): (len(g4a_s) + len(g4a_s2))]
b42 = b4all[len(g4a_s): (len(g4a_s) + len(g4a_s2))]
for a, b, nm in zip(a42, b42, g4a_s2):
    pep_loc.iloc[a, b] = nm
# print(pep_loc)


# 剩余元素填充
# 以中心为起点，逆序定义排序规则
acall = [4, 4, 5, 5, 5, 4, 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, 6, 5, 4, 3, 2, 2, 2, 2,
         2, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1,
         1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]

bcall = [4, 5, 5, 4, 3, 3, 3, 4, 5, 6, 6, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 3, 4, 5,
         6, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6,
         7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1]

# 整理父集 gaa
# g1a,g2a,g3a,g4a 整合为一个 df，去重，逐列排序（亲和力高优先），高亮填在中间，
# 避免填充边角时，高亮的剩余元素掩盖原先（数量可能较少的）指向性的边角元素
gaa_df = pd.concat([g1p_tb, g2p_tb, g3p_tb, g4p_tb])
# print(gaa_df.head())
# print(gaa_df.shape)           # (169, 2)  # 总共只有 100 个，说明有大量重复项

gaa_nd = gaa_df.drop_duplicates(subset="pep_name")
# gaa_nd.to_csv("tst.csv", index=False)

# 判断指定位置元素为 0 才填充（0是最初空板上的，说明没有被占用）
gaa = gaa_nd["pep_name"]
chajia = set(gaa) - set(sctr)  # 获得并验证 chaji of all，58 个差集元素
gaa_s = list(gaa)
for i in chajia:  # 最终的 s13 共享元素，带排序，9
    gaa_s.remove(i)
# print(len(gaa_s))   # 目标集 32 个元素
# 判断指定元素非零，需要重用当前的 gaa_s 肽名
i = 0
while i < len(acall):  # 以坐标列表为上限
    if pep_loc.iloc[acall[i], bcall[i]] != 0:  # 目标位置不为 0 则跳过该坐标
        i = i + 1
    elif pep_loc.iloc[acall[i], bcall[i]] == 0:
        len_gaas = len(gaa_s)
        if len_gaas:  # 以肽集合被取空为终止条件，break
            pep_loc.iloc[acall[i], bcall[i]] = gaa_s.pop(0)  # 每次取出第一个元素
            i = i + 1
        else:
            break
# print(pep_loc)  # 剩余 10 个位置，说明 90 个有亲和力目标均已放置完成

# 绘制格子化的布局图像（非矩阵）
# # 创建一个10x10的白色背景图像
# canvas = np.ones((10, 10, 3))
# # 绘制格子图像
# plt.figure(figsize=(3, 3))
# plt.axis('off')
# plt.imshow(canvas)
# # 绘制黑色格子线条
# for i in range(11):
#     plt.plot([-0.5, 9.5], [i - 0.5, i - 0.5], color='black', linewidth=1)
#     plt.plot([i - 0.5, i - 0.5], [-0.5, 9.5], color='black', linewidth=1)
# # 将字符串填入格子中
# for i in range(10):
#     for j in range(10):
#         text = pep_loc[i][j]
#         plt.text(j, i, text, fontsize=5, va='center', ha='center')
# # plt.show()
# plt.savefig("b.pep_loc1.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("b.pep_loc1.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02)

# pep_loc.to_csv("b.pep_buju.csv", index=False, header=None)
# 注：活性在 60 以下的还有 10 个元素，认为其活性不足以用来指示分类，可以不填入阵列中


# ===== Part 4 设计测试数据计算亲和力矩阵 =====================
# 在 DF 中依次取值（肽别名），若是非 0 字符，则在 tst 测试列表的 0 列中查找，
# 若找到则取 1 列中对应的值，将该值赋值给 DF 的当前位置，将别名替换为亲和力
# 获得的亲和力矩阵，绘制热图，矩阵或热图用于训练

# pep_loc = pd.read_csv("b.pep_buju.csv", header=None)
tst = pd.read_csv("b.pep_ceshi_list.csv", header=None)
# print(type(pep_loc))    # df
i = 0
while i < pep_loc.shape[0]:
    j = 0
    while j < pep_loc.shape[1]:
        if pep_loc.iloc[i, j] != 0:  # 非 0，在 tst 中查找并赋值
            lckey = pep_loc.iloc[i, j]  # 获取该位置的对象 str
            # lcval = list(pep_loc.iloc[i,j])[0]
            pep_loc.iloc[i, j] = list(tst[tst[0] == lckey][1])[0]
            # pep_loc.iloc[i, j] = tst.loc[tst[0] == lckey, 1].values[0]
        elif pep_loc.iloc[i, j] == 0:
            pass
        j = j + 1
    i = i + 1
# print(pep_loc)
# 将数据表转成矩阵形式，仍有空白，但可以保存矩阵使用
# pep_loc.to_csv("b.pep_ceshi_matrix.csv", index=False, header=None)


# 读入亲和力矩阵
# 查找矩阵中的空白元素，并使用列均值替换（即不用单独生成）
ceshi_m = pd.read_csv("b.pep_ceshi_matrix.csv", header=None)
i = 0
while i < ceshi_m.shape[0]:
    j = 0
    while j < ceshi_m.shape[1]:
        if ceshi_m.iloc[i, j] == 0:
            ceshi_m.iloc[i, j] = round(ceshi_m.mean()[j])
        j = j + 1
    i = i + 1
# print(ceshi_m)
# print(ceshi_m.mean())  # 均值是按列求得的
# 最终获得的 ceshi 对象就是完整的亲和力矩阵（包含均值替换的部分数值）


# Plot, annot 显示数值注释，fmt 指定显示类型为 d 整数 .1f 一位浮点，
# cbar 定义 color bar 是否显示
plt.figure(figsize=(2, 2))
sns.heatmap(ceshi_m, annot=True, annot_kws={"size": 4}, fmt="d", cmap="Greys",
            linewidths=0, cbar=False)
plt.xticks([])  # 手动定义轴标签为空
plt.yticks([])
plt.tight_layout()
# 输出时需要注意 pad_inches 来约束，否则会默认加上白框
# plt.savefig("b.pep_res_plot.pdf", bbox_inches='tight', pad_inches=0)
plt.savefig("b.pep_res_plot.jpg", dpi=300, bbox_inches="tight",
            pad_inches=0)  # 不保存则不必
# plt.show()


# ===== Part 5 加边框与定位角 ===============================
# 定位角在纯计算阶段暂不需要增加
# 有共有的肽集合，可以用作三个定位角的标注（任何目标气体的出现都高亮）
# hl = g1p & g2p & g3p & g4p  # {'pep10', 'pep22', 'pep69'}
# print(hl)


# ===== Part 6 自定义肽数据库和测试数据 =======================
# 定义肽库（Pandas），别名（编号）、肽名（数据框编号）、气体亲和力*4（方便以 -4 取出）
# 就是最开始定义的表格（真实数据需要整理为同样的表格，后面补充）
# 在项目 Part 1 中使用
