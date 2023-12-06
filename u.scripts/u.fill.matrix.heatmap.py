import numpy as np
import matplotlib.pyplot as plt

# 填充芯片，获得 matrix 矩阵并绘制热图
# 包括无混合插值和双立方混合插值两种插值算法

# 创建10*10的矩阵
matrix = np.zeros((10, 10))

# 填充区域
matrix[:3, :3] = np.random.randint(60, 98, size=(3, 3))

matrix[3:5, :3] = np.random.randint(50, 75, size=(2, 3))
matrix[:3, 3:5] = np.random.randint(50, 75, size=(3, 2))

matrix[5:7, :3] = np.random.randint(30, 60, size=(2, 3))
matrix[3:5, 3:5] = np.random.randint(30, 60, size=(2, 2))
matrix[:3, 5:7] = np.random.randint(30, 60, size=(3, 2))

matrix[7:10, :3] = np.random.randint(10, 40, size=(3, 3))
matrix[5:7, 3:5] = np.random.randint(10, 40, size=(2, 2))
matrix[3:5, 5:7] = np.random.randint(10, 40, size=(2, 2))
matrix[:3, 7:10] = np.random.randint(10, 40, size=(3, 3))

matrix[7:10, 3:5] = np.random.randint(10, 40, size=(3, 2))
matrix[5:7, 5:7] = np.random.randint(10, 40, size=(2, 2))
matrix[3:5, 7:10] = np.random.randint(10, 40, size=(2, 3))

matrix[5:7, 7:10] = np.random.randint(10, 40, size=(2, 3))
matrix[7:10, 5:7] = np.random.randint(10, 40, size=(3, 2))

matrix[7:10, 7:10] = np.random.randint(10, 40, size=(3, 3))

# imshow 绘制热图
plt.imshow(matrix, cmap='Greys', interpolation='nearest')
# 无混合插值 nearest, 双立方混合插值 bicubic
plt.xticks([])  # 手动定义轴标签为空
plt.yticks([])
plt.axis('off')
plt.tight_layout()
plt.show()
