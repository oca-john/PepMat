import numpy as np
import matplotlib.pyplot as plt

# 绘制格子线条，并加文本注释
# 将刻度线调整到轴内侧

# 创建一个10x10的全0矩阵
data = np.zeros((10, 10))

# 根据数据创建热图，显示格子线
plt.imshow(data, cmap='Greys', interpolation='nearest',

           extent=[0, 10, 0, 10], aspect='auto')

# 添加注释信息
for i in range(len(data)):
    for j in range(len(data[i])):
        plt.text(j + 0.5, i + 0.5, 'String',
                 ha='center', va='center', fontsize=6)

# 显示内部的格子线
plt.grid(color='white', linestyle='-', linewidth=1)
# 设置刻度线的位置
plt.xticks(np.arange(0, 10, 1), [])
plt.yticks(np.arange(0, 10, 1), [])
plt.tick_params(axis='both', which='both', direction='in')

# 显示热图
plt.colorbar()
plt.show()
