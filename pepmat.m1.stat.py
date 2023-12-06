import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tracemalloc
import time

# 此程序使用的数据文件是根据生成的数据反解出来的亲和力数据reor，总体分布与实验测试结果
# 获得的亲和力测试数据表相同。
# 采用统计学方法，统计每列中表示各组的行范围中数值的均值，关注其中处于对角线的四个数值，
# 它们代表了所在气体分组的亲和力强度，判断数值的绝对值大小，决定当次测验包含的气体数目，
# 之后根据气体数目决定计算公式（不包含的气体，系数修改为0）


tracemalloc.start()

start_time = time.time()  # 获取当前时间

# 用正则表达式过滤出reor文件名
pattern = re.compile(r'reor')
folder_path = 'dataset-06h/dats'
file_names = os.listdir(folder_path)
filtered_file_names = [file_name for file_name in file_names if
                       pattern.search(file_name)]
filtered_file_names.sort(reverse=False)
# print(len(filtered_file_names))  # 600, 共15种模式，每种4个，一共600个样本

# 读入生成时同步分配的标签
val_folder_path = 'dataset-06h/tags'
val_file_names = os.listdir(val_folder_path)
val_file_names.sort(reverse=False)
# print(len(file_names))  # 600, 共15种模式，每种40个，一共600个样本

dat_c_merge = []  # 数据
dat_v_merge = []  # 数据
cls_merge = []  # 从名称中读入的分类标签
val_merge = []  # 统一生成的分块亲和力均值标签（相对亲和力）
# i = 1

# 读入reor文件，分别按统计方法计算类别和相对含量，解析数据标签
for file_name, val_file in zip(filtered_file_names, val_file_names):
    file_path = os.path.join(folder_path, file_name)  # 拼接文件路径
    val_path = os.path.join(val_folder_path, val_file)

    # 文件名中解析的是分配的标签
    filename = file_name
    pattern = r"gas_([1-4]{1,4})_reor"
    match = re.search(pattern, filename)
    match_list = list(match.group(1))  # 是其中匹配的数字部分[求长度作为气体类别数]
    match_list = [int(char) for char in match_list]
    # print("There is(are)", len(match_list), "kinds of gas(es).")  # 气体数
    # print("This is the gases list:", match_list)  # 包含的气体列表
    # for g in match_list:
    #     print("This is the gas:", g)  # 逐个打印
    cls_merge.append(match_list)

    val_ls = pd.read_csv(val_path, index_col=None, header=None)
    val_ls = val_ls.values.tolist()[0]
    # print(val_ls, type(val_ls))
    val_merge.append(val_ls)

    # 读入文件，统计计算每列对应行区间的均值
    da = pd.read_csv(file_path, index_col=0, header=0)
    # print(da.shape, type(da))
    gs1 = da.iloc[0:25, 0].mean().round(2)
    gs2 = da.iloc[25:50, 1].mean().round(2)
    gs3 = da.iloc[50:75, 2].mean().round(2)
    gs4 = da.iloc[75:100, 3].mean().round(2)
    gss = [gs1, gs2, gs3, gs4]  # 规整统计计算的结果

    # 输出1：仅报告相对含量（无效项为0）
    # 均值比较，用大值数量判定数量，用大值比例计算相对含量
    # 均值大于50认为存在该气体
    # 将上述亲和力均值合并为list，然后将小于50的替换为空
    gss_t = ["" if x < 57 else x for x in gss]  # 替换小于50的数据为空
    gss_na = [x for x in gss_t if x != ""]  # 非空元素
    gss_in = [i + 1 for i, x in enumerate(gss_t) if x]  # 非空元素索引号(+1)
    # print(len(gss_in), gss_in)  # 第二种方式（只报告类别，可自行计数）

    # gss_sub = [gss[n-1] for n in gss_in]
    # # print(gss_sub)  # 获得有效的气体子集
    # gss_per = [(x*100/sum(gss_sub)).round(2) for x in gss_sub]
    # # print(gss_per)  # 按照存在的气体类别计算百分比（相对含量）
    # # print(len(gss_in), gss_in, gss_per)  # 整合后的报告信息
    dat_c_merge.append(gss_in)  # 直接拼合后是list

    # 输出2：报告亲和力均值列表
    oup = pd.DataFrame(gss).T
    oup = oup.values.tolist()[0]  # 转列表
    dat_v_merge.append(oup)
    # i += 1
    # if i >= 63:
    #     break

# 自建数据集（1行数据对应4个特征数值）
dat_cls = dat_c_merge
# print(dat_cls)
# print(len(dat_cls), type(dat_cls))
# print(dat_cls[0], type(dat_cls[0]))  # 第一行包含100个特征值

dat_val = dat_v_merge
# print(np.array(dat_val).shape)
# print(len(dat_val), type(dat_val))
# print(dat_val[0], type(dat_val[0]))

# 从标题提取的分类标签
cls = cls_merge
# print(cls)
# print(len(cls), type(cls))
# print(cls[0], type(cls[0]))  # 第一个元素的值

# 统一生成的相对含量标签
val = val_merge
# print(np.array(val).shape)
# print(len(val), type(val))
# print(val[0], type(val[0]))


# 从原始数据计算4个亲和力值，然后用这个计算分类结果和亲和力结果，与参考标签进行对照算正确率

# 分类结果评估
# 实际计算获得的分类结果与标签相比，正确率即预测准确率
# 划分为4个子集，但只用测试部分
dat_cls_train, dat_cls_tst, cls_train, cls_tst = train_test_split(dat_cls,
                                                                  cls,
                                                                  test_size=0.2,
                                                                  random_state=11)
count_right = 0
total_rows = len(dat_cls_tst)

for i in range(total_rows):
    if dat_cls_tst[i][:4] == cls_tst[i][:4]:  # 忽略掉多余的数字
        count_right += 1

percentage_right_rows = (count_right / total_rows).__round__(4)
print("Accuracy:", percentage_right_rows)

# 回归结果评估
# 将数据转为4列
# 每列单独计算R2分数和MSE，之后计算4列的平均值
# 划分为4个子集，但只用测试部分
dat_val_train, dat_val_tst, val_train, val_tst = train_test_split(dat_val,
                                                                  val,
                                                                  test_size=0.2,
                                                                  random_state=22)
# print(type(dat_val_tst), np.array(dat_val_tst).shape)
# 将4列数据分别提取为4个单独的列表
sub_dat1 = np.array(dat_val_tst)[:, 0].tolist()
sub_dat2 = np.array(dat_val_tst)[:, 1].tolist()
sub_dat3 = np.array(dat_val_tst)[:, 2].tolist()
sub_dat4 = np.array(dat_val_tst)[:, 3].tolist()

sub_val1 = np.array(val_tst)[:, 0].tolist()
sub_val2 = np.array(val_tst)[:, 1].tolist()
sub_val3 = np.array(val_tst)[:, 2].tolist()
sub_val4 = np.array(val_tst)[:, 3].tolist()

r2_1 = r2_score(sub_val1, sub_dat1).round(4)
r2_2 = r2_score(sub_val2, sub_dat2).round(4)
r2_3 = r2_score(sub_val3, sub_dat3).round(4)
r2_4 = r2_score(sub_val4, sub_dat4).round(4)

r2 = ((abs(r2_1) + abs(r2_2) + abs(r2_3) + abs(r2_4)) / 4).__round__(4)
# print(r2_1, r2_2, r2_3, r2_4)
print("R2 Score:", r2)

mse_1 = mean_squared_error(sub_val1, sub_dat1).round(4)
mse_2 = mean_squared_error(sub_val2, sub_dat2).round(4)
mse_3 = mean_squared_error(sub_val3, sub_dat3).round(4)
mse_4 = mean_squared_error(sub_val4, sub_dat4).round(4)

mse = ((mse_1 + mse_2 + mse_3 + mse_4) / 4).__round__(4)
# print(mse_1, mse_2, mse_3, mse_4)
print("Mean Squared Error:", mse)


current_memory, peak_memory = tracemalloc.get_traced_memory()
print(
    f"Current memory usage: {(current_memory / 1024 / 1024).__round__(4)} MB")
print(f"Peak memory usage: {(peak_memory / 1024 / 1024).__round__(4)} MB")
tracemalloc.stop()

end_time = time.time()  # 获取当前时间
run_time = end_time - start_time  # 计算程序运行时间
run_time_rounded = round(run_time, 4)  # 保留三位小数
print("The program ran for: ", run_time_rounded, "S")
