
import tensorflow as tf
import numpy as np

# 生成随机数据
data1 = np.random.random((5, 5)).round(2)
data2 = np.random.random((5, 5)).round(2)

# 转换数据为张量
input1 = tf.convert_to_tensor(data1)
input2 = tf.convert_to_tensor(data2)
print(input1)  # shape=(5, 5)
print(input2)  # shape=(5, 5)

# 使用 concatenate 合并两个输入张量
merge_layer = tf.keras.layers.concatenate([input1, input2], axis=1)  # 横向合并
# shape=(5, 10)
# merge_layer = tf.keras.layers.concatenate([input1, input2], axis=0)  # 纵向合并
# shape=(10, 5)

# 打印合并后的张量
print(merge_layer)
