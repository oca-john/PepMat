import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import random
import os

# 由于用到了最新的语法结构，要求Python版本大于3.10
# 此程序用于构建人工数据集，包含4中组分15种组合的所有类型
# 按照训练和测试需要的数据量生成数据
# 初步的最大需求量设计为 6000 组数据（包括数据矩阵和无边框的黑白热图）


# >>> 准备：坐标系、数据序列、填充函数 //////////////////////////////////////////////
# 分区生成仅需要4个区域，一共15种组合方式（共生成6000组，每种组合生成400组数据）
# 有混合情况的时候，需要额外注意判断，先判断第二种气体存在，才能考虑其对其它分组中同一气体的影响

# 坐标系统沿用 Beta 阶段的设计
a1 = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
b1 = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 4, 3, 2, 4, 3, 4]
a2 = b1
b2 = [9 - x for x in a1]
a3 = [9 - x for x in a1]
b3 = [9 - y for y in b1]
a4 = [9 - x for x in a1]
b4 = b1


# HighAffi 区的生成函数（负责水平在5或4的15个数据，和水平在3或2的10个数据）
# 1 - 41 - 56 - 71 - 86 - 100
# 容错范围设为 (-4, +4)
# 一次应用生成25个数据序列
def hag():  # 不需要参数，只要调用，就返回序列化的25个数据，后由填充次序决定位置
    hsq = [random.randint(71, 100) for _ in range(10)]
    hsq = hsq + [random.randint(56, 86) for _ in range(7)]
    hsq = hsq + [random.randint(41, 71) for _ in range(8)]
    # hsq = sorted(hsq, reverse=True)  # 去掉排序，改成多次生成、合并列表、依次填充
    return hsq


# Blank 区的生成函数（负责水平在1的75个数据）
# 一次应用生成25个空白区数据序列
def lag():
    lsq = [random.randint(1, 45) for _ in range(25)]
    # lsq = sorted(lsq, reverse=True)
    return lsq


# 填充函数
# 由于4个区域填充次序可变，因此填充函数只包含顺序填入的25个数值
# 坐标（区位）信息和数据都需要用户指定
# 并在调用4次后，拼合为完整的布局设计
# 调用时用到的 mtx 此时还未创建，函数内对 mtx 的操作会引起报错，用形参占位即可
def fil(x, y, f, mtx=None):
    for i in range(25):
        row = x[i]
        col = y[i]
        mtx[row][col] = f()[i]


def rannd(loc, size):
    # 设置随机种子
    np.random.seed(13)
    # 生成均值为70的25个随机数
    random_numbers = np.random.normal(loc=loc, size=size)
    random_numbers = np.round(random_numbers, 3)
    return random_numbers


# 读入归一化版本的数据，获取索引号（多肽列表）和列名（气体列表）
ref_affi = pd.read_csv("a.qinhe_norm_slec.csv", header=0, index_col=0)
ind_lis = ref_affi.index
# ind_lis = ind_lis.tolist()  # set_index()设置索引时需要保持索引格式
col_lis = ref_affi.columns
col_lis = col_lis.tolist()
# print(ind_lis)
# print(col_lis)


# >>> 数值矩阵生成 //////////////////////////////////////////////////////////////
# 区分15种情况的构建函数
def gen_gas(ct):
    match ct:  # match-case 语法需要 python 3.10 及更新的版本
        case 1:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)  # 调用时的形参需要传入
            fil(a2, b2, lag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, lag, mtx)
            g1 = pd.DataFrame(mtx)
            return g1
        case 2:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, lag, mtx)
            g2 = pd.DataFrame(mtx)
            return g2
        case 3:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, lag, mtx)
            g3 = pd.DataFrame(mtx)
            return g3
        case 4:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, hag, mtx)
            g4 = pd.DataFrame(mtx)
            return g4
        case 12:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, lag, mtx)
            g12 = pd.DataFrame(mtx)
            return g12
        case 13:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, lag, mtx)
            g13 = pd.DataFrame(mtx)
            return g13
        case 14:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, hag, mtx)
            g14 = pd.DataFrame(mtx)
            return g14
        case 23:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, lag, mtx)
            g23 = pd.DataFrame(mtx)
            return g23
        case 24:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, hag, mtx)
            g24 = pd.DataFrame(mtx)
            return g24
        case 34:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, hag, mtx)
            g34 = pd.DataFrame(mtx)
            return g34
        case 123:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, lag, mtx)
            g123 = pd.DataFrame(mtx)
            return g123
        case 124:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, lag, mtx)
            fil(a4, b4, hag, mtx)
            g124 = pd.DataFrame(mtx)
            return g124
        case 134:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, lag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, hag, mtx)
            g134 = pd.DataFrame(mtx)
            return g134
        case 234:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, lag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, hag, mtx)
            g234 = pd.DataFrame(mtx)
            return g234
        case 1234:
            mtx = [["" for _ in range(10)] for _ in range(10)]
            fil(a1, b1, hag, mtx)
            fil(a2, b2, hag, mtx)
            fil(a3, b3, hag, mtx)
            fil(a4, b4, hag, mtx)
            g1234 = pd.DataFrame(mtx)
            return g1234
        case _:
            return "Not gen any gase!"


# print(gen_gas(1))  # 测试一次完整的生成（需手动指定方案）


# >>> 数值矩阵生成 //////////////////////////////////////////////////////////////
# 构建调用函数，基于 gen_gas() 函数生成数据，并格式化
# 包含方案名、循环次数
# 数据生成并自动命名（DL_CD）、数据打乱（DL）、数据展平（ML）、数据反解（ST）
# 数据可视化、图形自动命名
tms = pd.read_csv('a.qinhe_tms_mtx.csv', header=None)


# print(tms)


# 根据用户指定的方案名，生成的数据量生成数据
def gen_form(sche, tims):
    for j in range(tims):
        # 输出方案名（组成），作为这组数据的真值
        # sche = sche
        # 自动归档和子文件夹命名
        subdir1 = "dats"
        os.makedirs(subdir1, exist_ok=True)
        # subdir2 = "imgs"
        # os.makedirs(subdir2, exist_ok=True)
        subdir3 = "tags"
        os.makedirs(subdir3, exist_ok=True)
        # 合成文件名（包含方案名和循环次数）
        dat_orig0 = f'gas_{sche}_orig_{j}.csv'
        # img_orig0 = f'gas_{sche}_orig_{j}.jpg'
        tag0 = f'gas_{sche}_tag_{j}.csv'
        gn_orig = os.path.join(subdir1, dat_orig0)
        # img_orig = os.path.join(subdir2, img_orig0)
        tag = os.path.join(subdir3, tag0)

        # 生成数据（DL_CD）
        gn = gen_gas(sche)
        # print(gn)
        # 将数据保存为csv文件和图像文件
        gn.to_csv(gn_orig, index=False, header=False)
        # imshow 绘制热图，无混合插值 nearest, 双立方混合插值 bicubic
        # plt.figure(figsize=(2, 2))
        # plt.imshow(gn, cmap='Reds', interpolation='nearest')
        # plt.axis('off')
        # plt.savefig(img_orig, dpi=300, bbox_inches="tight", pad_inches=0)
        # plt.close()
        # plt.show()

        # 生成统一的 tags 文件
        q1 = gn.iloc[0:5, 0:5].mean().mean().round()
        q2 = gn.iloc[0:5, 5:10].mean().mean().round()
        q3 = gn.iloc[5:10, 5:10].mean().mean().round()
        q4 = gn.iloc[5:10, 0:5].mean().mean().round()
        # print(q_merg)
        q_merg = pd.DataFrame([q1, q2, q3, q4]).T
        q_merg.to_csv(tag, index=False, header=False)

        # 数据打乱（DL）
        # 合成文件名（包含方案名和循环次数）
        dat_shuf0 = f'gas_{sche}_shuf_{j}.csv'
        # img_shuf0 = f'gas_{sche}_shuf_{j}.jpg'
        dat_shuf = os.path.join(subdir1, dat_shuf0)
        # img_shuf = os.path.join(subdir2, img_shuf0)
        # DL 和 ML 部分的结果需要用一个固定的数组验证一下，多次打乱是否规则相同
        # 指定随机状态保证所有数据在同一套规则下被打乱，以方便模式学习，否则将毫无意义
        gn_shuf = gn.sample(axis=0, frac=1, random_state=42)  # 按行打乱
        gn_shuf = gn_shuf.sample(axis=1, frac=1, random_state=42)  # 按列打乱
        # print(gn)  # 原有数据框未修改
        # print(gn_shuf)  # 打乱的数据给另一个对象
        # 将数据保存为csv文件和图像文件
        gn_shuf.to_csv(dat_shuf, index=False, header=False)
        # plt.figure(figsize=(2, 2))
        # plt.imshow(gn_shuf, cmap='Reds', interpolation='nearest')
        # plt.axis('off')
        # plt.savefig(img_shuf, dpi=300, bbox_inches="tight", pad_inches=0)
        # plt.close()
        # plt.show()

        # 数据展平（ML）随机指纹谱
        # 合成文件名（包含方案名和循环次数）
        dat_flat0 = f'gas_{sche}_flat_{j}.csv'
        # img_flat0 = f'gas_{sche}_flat_{j}.jpg'
        dat_flat = os.path.join(subdir1, dat_flat0)
        # img_flat = os.path.join(subdir2, img_flat0)
        # 展平规则明确，不会引起不同数据之间的差异
        gn_flat = gn_shuf.values.reshape(100, 1)
        gn_flat = gn_flat.T
        # print(type(gn_flat))  # ndarray
        gn_flat = pd.DataFrame(gn_flat)
        # print(gn_flat)
        # 将数据保存为csv文件和图像文件
        gn_flat.to_csv(dat_flat, index=False, header=False)
        # plt.figure()
        # plt.imshow(gn_flat, cmap='Reds', interpolation='nearest')
        # plt.axis('off')
        # plt.savefig(img_flat, dpi=300, bbox_inches="tight", pad_inches=0)
        # plt.close()
        # plt.show()

        # 数据反解（ST）亲和力均值矩阵
        # 合成文件名（包含方案名和循环次数）
        dat_reor0 = f'gas_{sche}_reor_{j}.csv'
        dat_reor = os.path.join(subdir1, dat_reor0)
        # 计算四块区域的均值，按照比率矩阵还原为真实亲和力数值，取三位小数
        gn_mean1 = gn.iloc[0:5, 0:5].mean().mean()
        gn_mean2 = gn.iloc[0:5, 5:10].mean().mean()
        gn_mean3 = gn.iloc[5:10, 5:10].mean().mean()
        gn_mean4 = gn.iloc[5:10, 0:5].mean().mean()
        # 芯片区块中25种多肽在单次实验中会与所有气体分子接触，其表现代表了整行的亲和力
        # 芯片设计为了平衡4个区块，将亲和力强度进行了调整，以公平地区分气体类别
        # 反解本质是按照芯片设计之初的亲和力相对值，将实验获得的亲和力还原为平衡之前的数值
        # 之后根据区块的平均值，按照指定的随机种子生成区块内的数值分布，然后重构为列表样式
        # 实际的统计计算过程，则是需要重新计算各部分的统计特征，并比较大小、做出判断
        # 其他区域反解依赖相对于本行最大值的比例（四个代表区块的比例是相对0-0而言的）
        gn_rt = [[tms.iloc[0, 0] / tms.iloc[0, :].max(),
                  tms.iloc[0, 1] / tms.iloc[0, :].max(),
                  tms.iloc[0, 2] / tms.iloc[0, :].max(),
                  tms.iloc[0, 3] / tms.iloc[0, :].max()],
                 [tms.iloc[1, 0] / tms.iloc[1, :].max(),
                  tms.iloc[1, 1] / tms.iloc[1, :].max(),
                  tms.iloc[1, 2] / tms.iloc[1, :].max(),
                  tms.iloc[1, 3] / tms.iloc[1, :].max()],
                 [tms.iloc[2, 0] / tms.iloc[2, :].max(),
                  tms.iloc[2, 1] / tms.iloc[2, :].max(),
                  tms.iloc[2, 2] / tms.iloc[2, :].max(),
                  tms.iloc[2, 3] / tms.iloc[2, :].max()],
                 [tms.iloc[3, 0] / tms.iloc[3, :].max(),
                  tms.iloc[3, 1] / tms.iloc[3, :].max(),
                  tms.iloc[3, 2] / tms.iloc[3, :].max(),
                  tms.iloc[3, 3] / tms.iloc[3, :].max()]]
        gn_rt = np.array(gn_rt).round(3)
        # print(gn_rt)
        # 逐行按相对量计算均值，获得16个亲和力均值
        # 将16个均值按照4行的参考比例还原回原始比例（乘以各行的缩放比例）
        gn_reor = [[gn_rt[0, 0] * gn_mean1 * tms.iloc[0, 0],
                    gn_rt[0, 1] * gn_mean1 * tms.iloc[0, 0],
                    gn_rt[0, 2] * gn_mean1 * tms.iloc[0, 0],
                    gn_rt[0, 3] * gn_mean1 * tms.iloc[0, 0]],
                   [gn_rt[1, 0] * gn_mean2 * tms.iloc[1, 1],
                    gn_rt[1, 1] * gn_mean2 * tms.iloc[1, 1],
                    gn_rt[1, 2] * gn_mean2 * tms.iloc[1, 1],
                    gn_rt[1, 3] * gn_mean2 * tms.iloc[1, 1]],
                   [gn_rt[2, 0] * gn_mean3 * tms.iloc[2, 2],
                    gn_rt[2, 1] * gn_mean3 * tms.iloc[2, 2],
                    gn_rt[2, 2] * gn_mean3 * tms.iloc[2, 2],
                    gn_rt[2, 3] * gn_mean3 * tms.iloc[2, 2]],
                   [gn_rt[3, 0] * gn_mean4 * tms.iloc[3, 3],
                    gn_rt[3, 1] * gn_mean4 * tms.iloc[3, 3],
                    gn_rt[3, 2] * gn_mean4 * tms.iloc[3, 3],
                    gn_rt[3, 3] * gn_mean4 * tms.iloc[3, 3]]]
        gn_reor = np.array(gn_reor).round(3)
        # 按照还原后的亲和力均值，生成一个区域（25个值）的数值（固定、统一的随机种子）
        d11 = rannd(gn_reor[0, 0], 25)
        d12 = rannd(gn_reor[0, 1], 25)
        d13 = rannd(gn_reor[0, 2], 25)
        d14 = rannd(gn_reor[0, 3], 25)
        d1 = np.column_stack((d11, d12, d13, d14))
        d21 = rannd(gn_reor[1, 0], 25)
        d22 = rannd(gn_reor[1, 1], 25)
        d23 = rannd(gn_reor[1, 2], 25)
        d24 = rannd(gn_reor[1, 3], 25)
        d2 = np.column_stack((d21, d22, d23, d24))
        d31 = rannd(gn_reor[2, 0], 25)
        d32 = rannd(gn_reor[2, 1], 25)
        d33 = rannd(gn_reor[2, 2], 25)
        d34 = rannd(gn_reor[2, 3], 25)
        d3 = np.column_stack((d31, d32, d33, d34))
        d41 = rannd(gn_reor[3, 0], 25)
        d42 = rannd(gn_reor[3, 1], 25)
        d43 = rannd(gn_reor[3, 2], 25)
        d44 = rannd(gn_reor[3, 3], 25)
        d4 = np.column_stack((d41, d42, d43, d44))
        da = np.row_stack((d1, d2, d3, d4))
        # 数据保存
        gn_reor = pd.DataFrame(da)
        gn_reor = gn_reor.set_index(ind_lis)
        # print(gn_reor.shape)
        # print(gn_reor)
        gn_reor.to_csv(dat_reor, index=True, header=col_lis)


# 数据生成测试
# gen_form(13, 1)


# >>> 批量数值生成 //////////////////////////////////////////////////////////////
# 调用构建函数生成数据，指定方案名和循环次数（数据体量）
# 数据体量需要在6千组（600, 1800, 6000）
# 由于包含了15种模式，t的设置需要时15的整数倍
def gen_all(t):
    gen_form(1, int(t/15))
    gen_form(2, int(t/15))
    gen_form(3, int(t/15))
    gen_form(4, int(t/15))
    gen_form(12, int(t/15))
    gen_form(13, int(t/15))
    gen_form(14, int(t/15))
    gen_form(23, int(t/15))
    gen_form(24, int(t/15))
    gen_form(34, int(t/15))
    gen_form(123, int(t/15))
    gen_form(124, int(t/15))
    gen_form(134, int(t/15))
    gen_form(234, int(t/15))
    gen_form(1234, int(t/15))


gen_all(600)
