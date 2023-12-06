import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import tracemalloc
import time

# 在循环内用正则表达式从dats文件夹中读取flat数据文件，每个循环处理一个数据，输出一个报告文件
# 使用机器学习方法，可选的使用特征提取方法，或直接应用分类器，获得对一维特征的分类结果
# 含量需要使用模式识别进行映射，或回归建模
# 调用过程中，直接用模型给出分类结果、估计的含量信息


tracemalloc.start()

start_time = time.time()  # 获取当前时间

# 用正则表达式过滤出flat文件名
pattern = re.compile(r'flat')
folder_path = 'dataset-06h/dats'
file_names = os.listdir(folder_path)
filtered_file_names = [file_name for file_name in file_names if
                       pattern.search(file_name)]
filtered_file_names.sort(reverse=False)
# print(len(filtered_file_names))  # 600, 共15种模式，每种40个，一共600个样本

# 读入生成时同步分配的标签
val_folder_path = 'dataset-06h/tags'
val_file_names = os.listdir(val_folder_path)
val_file_names.sort(reverse=False)
# print(len(file_names))  # 600, 共15种模式，每种40个，一共600个样本


# def generate_list(data):
#     result = [0, 0, 0, 0]  # 初始列表
#     for num in data:
#         if 1 <= num <= 4:
#             result[num - 1] = 1  # 设置对应位置的数据为1
#     return result


dat_merge = []  # 数据
cls_merge = []  # 从名称中读入的分类标签
val_merge = []  # 统一生成的分块亲和力均值标签（相对亲和力）
# i = 1

# 读入flat文件，合并为数据集，解析数据标签
for file_name, val_file in zip(filtered_file_names, val_file_names):
    file_path = os.path.join(folder_path, file_name)  # 拼接文件路径
    val_path = os.path.join(val_folder_path, val_file)

    filename = file_name
    pattern = r"gas_([1-4]{1,4})_flat"
    match = re.search(pattern, filename)
    match_list = list(match.group(1))  # 是其中匹配的数字部分[求长度作为气体类别数]
    match_list = [int(char) for char in match_list]
    new_list = []
    for num in match_list:
        if num == 3:
            new_list.append(4)
        elif num == 4:
            new_list.append(8)
        else:
            new_list.append(num)
    cls_sum = sum(new_list)

    # print("There is(are)", len(match_list), "kinds of gas(es).")  # 气体数
    # print("This is the gases list:", new_list)  # 包含的气体列表
    # for g in match_list:
    #     print("This is the gas:", g)  # 逐个打印
    # 参考linux的权限设计，我们用 0001, 0010, 0100, 1000 表示四种分类
    # 转为十进制则是，1, 2, 4, 8，测试组合：
    # 1, 2, 4, 8
    # 3, 5, 9, 6, 10, 12
    # 7, 11, 13, 14
    # 15

    # cls_lst = generate_list(match_list)
    cls_merge.append(cls_sum)

    da = pd.read_csv(file_path, index_col=None, header=None)
    # print(da.shape, type(da))
    dat_merge.append(da)  # 直接拼合后是list
    # print(len(dat_merge), type(dat_merge))

    val_ls = pd.read_csv(val_path, index_col=None, header=None)
    val_ls = val_ls.values.tolist()[0]
    # print(val_ls, type(val_ls))
    val_merge.append(val_ls)
    # i += 1
    # if i >= 3:
    #     break

# dat_merge 为 list，每个元素表示一个样本，为 10*10 的 DataFrame
# print(len(dat_merge))  # 120 数据集长度（每个元素为一个样本）
# print(type(dat_merge))  # <class 'list'>（数据框组合成的列表）
# print(dat_merge[0].shape)  # 读出第一个元素（数据框）
# print(type(dat_merge[0]))  # <class 'pandas.core.frame.DataFrame'>

# 自建数据集（1行数据对应4个特征数值）
dat = dat_merge
dat = np.reshape(dat, (len(dat), 100))
# print(len(dat), type(dat))
# print(dat[0].shape, type(dat[0]))  # 第一行包含100个特征值

# 从标题提取的分类标签
cls = cls_merge
# cls = np.reshape(cls, (len(cls), 4))
cls = np.array(cls)
# print(len(cls), type(cls))
# print(cls[0], type(cls[0]))  # 第一个元素的值

# 统一生成的相对含量标签
val = val_merge
val = np.reshape(val, (len(val), 4))
# print(len(val), type(val))
# print(val[0], type(val[0]))


# 用数据集与标签进行学习能够获得分类模型
train_dat, test_dat, train_cls, test_cls = train_test_split(dat,
                                                            cls,
                                                            test_size=0.2,
                                                            random_state=11)

# 加载模型
model_m2_cls = joblib.load('models-saved/model_m2_cls.pkl')
# 测试模型
pred_cls = model_m2_cls.predict(test_dat)
accuracy = np.mean(pred_cls == test_cls)
print("Accuracy:", accuracy)  # 1.0


# 用数据集与含量标签进行学习能获得含量估计的回归模型
train_dat, test_dat, train_val, test_val = train_test_split(dat,
                                                            val,
                                                            test_size=0.2,
                                                            random_state=22)

# # 多输出回归模型 - 向量机（预测结果较差，最终用决策树）
# svr = SVR()
# model_m2_reg = MultiOutputRegressor(svr)
# # 训练
# model_m2_reg.fit(train_dat, train_val)
# # 预测
# y_pred = model_m2_reg.predict(test_dat)
# # 评估
# r2 = r2_score(test_val, y_pred).round(4)
# mse = mean_squared_error(test_val, y_pred).round(4)
# print("R2 Score:", r2)  # 0.2806
# print("Mean Squared Error:", mse)  # 401.547

# 加载模型
model_m2_reg = joblib.load('models-saved/model_m2_reg.pkl')
# 预测
y_pred = model_m2_reg.predict(test_dat)
# 评估
r2 = r2_score(test_val, y_pred).round(4)
mse = mean_squared_error(test_val, y_pred).round(4)
print("R2 Score:", r2)  # 0.9864 （越接近1，预测能力越好）
print("Mean Squared Error:", mse)  # 7.3854 （越小，预测结果越准）
# print(test_val.shape)  # (24, 4)
# print(y_pred.shape)  # (24, 4)


current_memory, peak_memory = tracemalloc.get_traced_memory()
print(
    f"Current memory usage: {(current_memory / 1024 / 1024).__round__(4)} MB")
print(f"Peak memory usage: {(peak_memory / 1024 / 1024).__round__(4)} MB")
tracemalloc.stop()

end_time = time.time()  # 获取当前时间
run_time = end_time - start_time  # 计算程序运行时间
run_time_rounded = round(run_time, 4)  # 保留三位小数
print("The program ran for: ", run_time_rounded, "S")
