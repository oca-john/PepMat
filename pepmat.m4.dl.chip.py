import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
import tracemalloc
import time

# 在循环内用正则表达式从dats文件夹中读取orig数据文件，每个循环处理一个数据，输出一个报告文件
# 使用深度学习方法，考虑到芯片设计的影响，采用分块处理策略
# 对于每个块，我们使用略小于区块的卷积核，目的是最快速的（不超过两层迭代）获得抽象特征
# 类别特征包含在分块的相对表现
# 含量信息包含在分块卷积计算的结果中
# 调用过程中，可以直接由模型给出分类结果（人类用户也很直观看出）、推测的含量（人类可继续口算）


tracemalloc.start()

start_time = time.time()  # 获取当前时间

# 用正则表达式过滤出orig文件名
pattern = re.compile(r'orig')
folder_path = 'dataset-18h/dats'
file_names = os.listdir(folder_path)
filtered_file_names = [file_name for file_name in file_names if
                       pattern.search(file_name)]
filtered_file_names.sort(reverse=False)
# print(len(filtered_file_names))  # 600, 共15种模式，每种40个，一共600个样本

# 读入生成时同步分配的标签
val_folder_path = 'dataset-18h/tags'
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

# 读入orig文件，合并为数据集，解析数据标签
for file_name, val_file in zip(filtered_file_names, val_file_names):
    file_path = os.path.join(folder_path, file_name)  # 拼接文件路径
    val_path = os.path.join(val_folder_path, val_file)

    filename = file_name
    pattern = r"gas_([1-4]{1,4})_orig"
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
    cls_sum = sum(new_list) - 1

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
    # cls_lst = [bool(nu) for nu in cls_lst]  # 转为布尔类型（可选）
    cls_merge.append(cls_sum)

    da = pd.read_csv(file_path, index_col=None, header=None)
    # print(da.shape, type(da[0]))
    dat_merge.append(da)  # 直接拼合后是list
    # print(len(dat_merge), type(dat_merge))

    val_ls = pd.read_csv(val_path, index_col=None, header=None)
    val_ls = val_ls.values.tolist()[0]
    # print(val_ls, type(val_ls[0]))
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
dat = np.reshape(dat, (len(dat), 10, 10))
# print(len(dat), type(dat))
# print(dat[0].shape, type(dat[0]))  # 第一行包含100个特征值

# 从标题提取的分类标签
cls = cls_merge
# cls = np.reshape(cls, (len(cls), 4))
cls = np.array(cls)  # 解析的标签不都是4个，不能reshape，使用单个数值是更合适的方案
# print(len(cls), type(cls))
# print(cls[0], type(cls[0]))  # 第一个元素的值

# 统一生成的相对含量标签
val = val_merge
val = np.reshape(val, (len(val), 4))
# print(len(val), type(val))
# print(val[0], type(val[0]))


# 用数据集与标签进行学习能够获得分类模型

# 将数据集分割为训练集和测试集
train_dat, test_dat, train_cls, test_cls = train_test_split(dat, cls,
                                                            test_size=0.2,
                                                            random_state=11)

# 定义输入层
input_layer = tf.keras.layers.Input(shape=(10, 10, 1))
# print(input_layer)  # (None, 10, 10, 1)

# 分割输入为4个子块
sub_block1 = tf.keras.layers.Lambda(lambda x: x[:, :5, :5, :])(input_layer)
sub_block2 = tf.keras.layers.Lambda(lambda x: x[:, :5, 5:, :])(input_layer)
sub_block3 = tf.keras.layers.Lambda(lambda x: x[:, 5:, 5:, :])(input_layer)
sub_block4 = tf.keras.layers.Lambda(lambda x: x[:, 5:, :5, :])(input_layer)
# print(sub_block1)  # (None, 5, 5, 1)

# 处理子块1
conv_layer_1_1 = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(4, 4),
                                        strides=(1, 1),
                                        padding='valid')(sub_block1)
# print(conv_layer_1_1)
# 处理子块2
conv_layer_1_2 = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(4, 4),
                                        strides=(1, 1),
                                        padding='valid')(sub_block2)
# print(conv_layer_1_2)
# 处理子块3
conv_layer_1_3 = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(4, 4),
                                        strides=(1, 1),
                                        padding='valid')(sub_block3)
# print(conv_layer_1_3)
# 处理子块4
conv_layer_1_4 = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=(4, 4),
                                        strides=(1, 1),
                                        padding='valid')(sub_block4)
# print(conv_layer_1_4)

# 合并4个池化结果
# concatenate 默认不会方形合并，所以后续的处理会在某个维度上尺寸不够用
# 可以考虑合并三次，先分别横向合并1,2和4,3，之后再纵向合并这两个块
merge_layer_2_1 = tf.keras.layers.concatenate([conv_layer_1_1,
                                               conv_layer_1_2],
                                              axis=2)  # 横向合并
# print(merge_layer_2_1)
merge_layer_2_2 = tf.keras.layers.concatenate([conv_layer_1_4,
                                               conv_layer_1_3],
                                              axis=2)  # 横向合并
# print(merge_layer_2_2)
merged_layer = tf.keras.layers.concatenate([merge_layer_2_1, merge_layer_2_2],
                                           axis=1)  # 纵向合并
# print(merged_layer)

# 处理合并后的特征
conv_layer_2 = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(2, 2),
                                      strides=(1, 1),
                                      padding='valid')(merged_layer)
conv_layer_3 = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='valid')(conv_layer_2)
pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                         strides=(1, 1))(conv_layer_3)

# 输出层
output_layer = tf.keras.layers.Flatten()(pool_layer_1)
output_layer = tf.keras.layers.Dense(units=15,
                                     activation='softmax')(output_layer)

# 定义模型
model_m4_cls = tf.keras.models.Model(inputs=input_layer,
                                     outputs=output_layer)
# model_m4_cls.summary()

# 编译模型
model_m4_cls.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

# 模型训练
train_cls = tf.expand_dims(train_cls, axis=1)
model_m4_cls.fit(train_dat, train_cls,
                 validation_data=(test_dat, test_cls),
                 epochs=10, batch_size=24)

# 保存模型
model_m4_cls.save('models-saved/model_m4_cls')

# 评估模型
score = model_m4_cls.evaluate(test_dat, test_cls, verbose=0)
print(f'Accuracy: {score[1]}')


# 用数据集与含量标签进行学习能获得含量估计的回归模型

# 划分训练集和测试集
train_dat, test_dat, train_val, test_val = train_test_split(dat, val,
                                                            test_size=0.2,
                                                            random_state=22)

# 定义输入层
input_layer = tf.keras.layers.Input(shape=(10, 10, 1))
# print(input_layer)  # (None, 10, 10, 1)

# 分割输入为4个子块
sub_block1 = tf.keras.layers.Lambda(lambda x: x[:, :5, :5, :])(input_layer)
sub_block2 = tf.keras.layers.Lambda(lambda x: x[:, :5, 5:, :])(input_layer)
sub_block3 = tf.keras.layers.Lambda(lambda x: x[:, 5:, 5:, :])(input_layer)
sub_block4 = tf.keras.layers.Lambda(lambda x: x[:, 5:, :5, :])(input_layer)
# print(sub_block1)  # (None, 5, 5, 1)

# 处理子块1
conv_layer_1_1 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(4, 4),
                                        activation='relu',
                                        strides=(1, 1),
                                        padding='valid')(sub_block1)
# print(conv_layer_1_1)
# 处理子块2
conv_layer_1_2 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(4, 4),
                                        activation='relu',
                                        strides=(1, 1),
                                        padding='valid')(sub_block2)
# print(conv_layer_1_2)
# 处理子块3
conv_layer_1_3 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(4, 4),
                                        activation='relu',
                                        strides=(1, 1),
                                        padding='valid')(sub_block3)
# print(conv_layer_1_3)
# 处理子块4
conv_layer_1_4 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(4, 4),
                                        activation='relu',
                                        strides=(1, 1),
                                        padding='valid')(sub_block4)
# print(conv_layer_1_4)

# 合并4个池化结果
# concatenate 默认不会方形合并，所以后续的处理会在某个维度上尺寸不够用
# 可以考虑合并三次，先分别横向合并1,2和4,3，之后再纵向合并这两个块
merge_layer_2_1 = tf.keras.layers.concatenate([conv_layer_1_1,
                                               conv_layer_1_2],
                                              axis=2)  # 横向合并
# print(merge_layer_2_1)
merge_layer_2_2 = tf.keras.layers.concatenate([conv_layer_1_4,
                                               conv_layer_1_3],
                                              axis=2)  # 横向合并
# print(merge_layer_2_2)
merged_layer = tf.keras.layers.concatenate([merge_layer_2_1, merge_layer_2_2],
                                           axis=1)  # 纵向合并
# print(merged_layer)

# 处理合并后的特征
conv_layer_2 = tf.keras.layers.Conv2D(filters=128,
                                      kernel_size=(2, 2),
                                      activation='relu',
                                      strides=(1, 1),
                                      padding='valid')(merged_layer)
conv_layer_3 = tf.keras.layers.Conv2D(filters=128,
                                      kernel_size=(1, 1),
                                      activation='relu',
                                      strides=(1, 1),
                                      padding='valid')(conv_layer_2)
pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                         strides=(1, 1))(conv_layer_3)

# 输出层
output_layer = tf.keras.layers.Flatten()(pool_layer_1)
output_layer = tf.keras.layers.Dense(units=4,
                                     activation='linear')(output_layer)

# 定义模型
model_m4_reg = tf.keras.models.Model(inputs=input_layer,
                                     outputs=output_layer)
# model_m4_reg.summary()

# 编译和训练
model_m4_reg.compile(loss='mse', optimizer='adam')
model_m4_reg.fit(train_dat, train_val, epochs=10, batch_size=24)
# 600 数据规模 -> 12
# 1800 数据规模 -> 24
# 6000 数据规模 -> 120

# 保存模型
model_m4_reg.save('models-saved/model_m4_reg')

# 模型评估
y_pred = model_m4_reg.predict(test_dat)
r2 = r2_score(test_val, y_pred).round(4)
print("R2 Score:", r2)  # 0.9924 （越接近1，预测能力越好）
mse = mean_squared_error(test_val, y_pred).round(4)
print("Mean Squared Error:", mse)  # 4.4876 （越小，预测结果越准）
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
