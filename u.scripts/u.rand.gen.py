import numpy as np

# 设置随机种子
np.random.seed(192)

# 生成均值为70，标准差为2的25个随机数
random_numbers = np.random.normal(loc=70, scale=13, size=25)
random_numbers = np.round(random_numbers, 3)

print(random_numbers)
