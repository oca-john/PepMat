import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 第三阶段
# 读入和复用第二阶段计算完成的集合信息
# 读入第一阶段确定好的气体集合（4）和多肽集合（100）
gas = pd.read_csv("b.gasname.csv", header=None)
gas = set(list(gas[0]))  # pd -> list -> set
pep = pd.read_csv("b.pepname.csv", header=None)
pep = set(list(pep[0]))

# 读入第二阶段的独享和两两亲和力的坐标和多肽，独享集合具有指示作用，
# 因此用于指定气体的指向性（类）和浓度（量）
# 设计 4 个等级指示，放入一个列表，生成 1-4 之间的随机数，从列表取出，
# 分别为其赋值（指定范围内的随机数）
# 独享亲和力（多肽列表）
s1 = pd.read_csv("sets/c.s1.csv", header=None)  # 操作 df，不转 set
s2 = pd.read_csv("sets/c.s2.csv", header=None)
s3 = pd.read_csv("sets/c.s3.csv", header=None)
s4 = pd.read_csv("sets/c.s4.csv", header=None)
# 两两亲和力（多肽列表）
s12 = pd.read_csv("sets/c.s12.csv", header=None)
s13 = pd.read_csv("sets/c.s13.csv", header=None)
s24 = pd.read_csv("sets/c.s24.csv", header=None)
s34 = pd.read_csv("sets/c.s34.csv", header=None)
# 其他亲和力（多肽列表）
sctr = pd.read_csv("sets/c.sctr.csv", header=None)
# 计算剩余的未被计算的空白坐标（对所有气体都不敏感的多肽（最初没通过任何气体的亲和力筛选），
# 不用记名，直接生成和赋值）
las_n = 100 - (len(s1) + len(s2) + len(s3) + len(s4) + len(s12) + len(s13)
               + len(s24) + len(s34) + len(sctr))
# print(las_n)  # 10，即表示有 10 个多肽是亲和力不够的
las = pd.DataFrame(np.zeros(10, object).reshape(10, 1))
# print(las.shape)    # (10, 1)


# 按照权重生成指定范围的随机数据，向上述列表中增加第二列，填充数值
# 设置四个方案 1-4，每个方案都设计对应的 l1 - l4 的赋值
# 定义函数，应用参数来生成数据
def data_4spec(ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8):
    lens1 = s1.shape[0]
    rand_vals = np.random.randint(ps1, ps2, size=lens1)  # 按行生成随机数
    s1[1] = rand_vals  # 将随机数添加到第二列
    # print(s1)
    lens2 = s2.shape[0]
    rand_vals = np.random.randint(ps3, ps4, size=lens2)
    s2[1] = rand_vals
    lens3 = s3.shape[0]
    rand_vals = np.random.randint(ps5, ps6, size=lens3)
    s3[1] = rand_vals
    lens4 = s4.shape[0]
    rand_vals = np.random.randint(ps7, ps8, size=lens4)
    s4[1] = rand_vals
    return s1, s2, s3, s4


def data_6norm():
    # 其他集合不具有指向性，不用区分
    lens12 = s12.shape[0]
    rand_vals = np.random.randint(20, 60, size=lens12)  # 按行生成随机数
    s12[1] = rand_vals  # 将随机数添加到第二列
    lens13 = s13.shape[0]
    rand_vals = np.random.randint(20, 60, size=lens13)
    s13[1] = rand_vals
    lens24 = s24.shape[0]
    rand_vals = np.random.randint(20, 60, size=lens24)
    s24[1] = rand_vals
    lens34 = s34.shape[0]
    rand_vals = np.random.randint(20, 60, size=lens34)
    s34[1] = rand_vals
    len_sctr = sctr.shape[0]
    rand_vals = np.random.randint(0, 40, size=len_sctr)
    sctr[1] = rand_vals
    rand_vals = np.random.randint(0, 20, size=las_n)
    las[1] = rand_vals
    # print(las)
    return s12, s13, s24, s34, sctr, las


def calc_matrix_and_plot():
    # 在 DF 中依次取值（肽别名），若是非 0 字符，则在 tst 测试列表的 0 列中查找，
    # 若找到则取 1 列中对应的值，将该值赋值给 DF 的当前位置，将别名替换为亲和力
    # 获得的亲和力矩阵，绘制热图，矩阵或热图用于训练
    pep_loc = pd.read_csv("b.pep_buju.csv", header=None)
    tst = pd.read_csv("c.pep_ceshi_list.csv", header=None)
    # print(type(pep_loc))    # df
    i = 0
    while i < pep_loc.shape[0]:
        j = 0
        while j < pep_loc.shape[1]:
            if pep_loc.iloc[i, j] != 0:  # 非 0，在 tst 中查找并赋值
                lckey = pep_loc.iloc[i, j]  # 获取该位置的对象 str
                # lcval = list(pep_loc.iloc[i,j])[0]
                pep_loc.iloc[i, j] = list(tst[tst[0] == lckey][1])[0]
            elif pep_loc.iloc[i, j] == 0:
                pass
            j = j + 1
        i = i + 1
    # print(pep_loc)
    # 将数据表转成矩阵形式，仍有空白，但可以保存矩阵使用
    # pep_loc.to_csv("c.pep_ceshi_matrix.csv", index=False, header=None)

    # 查找矩阵中的空白元素，并使用列均值替换（即不用单独生成）
    # ceshi = pd.read_csv("c.pep_ceshi_matrix.csv", header=None)
    ceshi_m = pep_loc
    i = 0
    while i < ceshi_m.shape[0]:
        j = 0
        while j < ceshi_m.shape[1]:
            if ceshi_m.iloc[i, j] == 0:
                ceshi_m.iloc[i, j] = round(ceshi_m.mean()[j])
            j = j + 1
        i = i + 1
    # 最终获得的 ceshi 对象就是完整的亲和力矩阵（包含均值替换的部分数值）

    # Plot, annot 显示数值注释，fmt 指定显示类型为 d 整数 .1f 一位浮点，
    # cbar 定义 color bar 是否显示
    plt.figure(figsize=(2, 2))
    sns.heatmap(ceshi_m, annot=False, fmt='d', cmap="Greys", linewidths=0,
                cbar=False)
    plt.xticks([])  # 手动定义轴标签为空
    plt.yticks([])
    plt.tight_layout()
    # 输出时需要注意 pad_inches 来约束，否则会默认加上白框
    # plt.savefig("c.pep_res_plot.pdf", dpi=300,
    # bbox_inches='tight', pad_inches=0)
    plt.savefig("c.pep_res_plot.png", dpi=300, bbox_inches='tight',
                pad_inches=0)  # 不保存则不必
    # plt.show()


# 气体1*100，生成参数组，应用参数产生随机数，整合随机数和气体名，保存数据，绘图并保存图像
for _ in range(100):
    ps1, ps2 = 60, 100
    ps3, ps4 = ps5, ps6 = ps7, ps8 = 30, 75
    # 生成指向性的前四项
    data_4spec(ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8)
    # 生成通用的后六项
    data_6norm()
    # 合并数据为单个对象
    ceshi_l = pd.concat([s1, s2, s3, s4, s12, s13, s24, s34,
                         sctr, las], axis=0)
    # print(ceshi)
    # 保存为文件，设初始值，递增命名
    # 创建数据子文件夹
    subdir1 = "datas_test"
    os.makedirs(subdir1, exist_ok=True)
    # 写入文件
    n = _
    filename = os.path.join(subdir1, "c.pep_ceshi_list\n.csv")
    ceshi_l.to_csv(filename, index=False, header=None)
    print("Saved datas to c.pep_ceshi_list file.")

###########################
import csv
# 定义基础文件名和起始索引
base_filename = "data"
name_n = 1
# 循环保存CSV文件
for i in range(10):
    filename = base_filename + "_" + str(name_n) + ".csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    index += 1
#########################

    # 根据表格，计算亲和力矩阵

    # 绘图并保存，设初始值，递增命名
    subdir2 = "images"
    os.makedirs(subdir2, exist_ok=True)
    filename = os.path.join(subdir2, 'plot_{}.png'.format(timestamp))
    plt.savefig(filename)

# # 气体2*100，生成参数组，应用参数产生随机数，整合随机数和气体名，保存数据，绘图并保存图像
# for _ in range(100):
#     ps3, ps4 = 60, 100
#     ps1, ps2 = ps5, ps6 = ps7, ps8 = 30, 75
#     data_4spec(ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8)  # 生成具有指向性的前四项数据
#
# # 气体3*100，生成参数组，应用参数产生随机数，整合随机数和气体名，保存数据，绘图并保存图像
# for _ in range(100):
#     ps5, ps6 = 60, 100
#     ps3, ps4 = ps1, ps2 = ps7, ps8 = 30, 75
#     data_4spec(ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8)  # 生成具有指向性的前四项数据
#
# # 气体4*100，生成参数组，应用参数产生随机数，整合随机数和气体名，保存数据，绘图并保存图像
# for _ in range(100):
#     ps7, ps8 = 60, 100
#     ps3, ps4 = ps5, ps6 = ps1, ps2 = 30, 75
#     data_4spec(ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8)  # 生成具有指向性的前四项数据
