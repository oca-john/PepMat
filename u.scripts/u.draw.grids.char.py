import numpy as np
import matplotlib.pyplot as plt

# 以 inshow 绘制格子线条，并按指定的文本注释
# 没有刻度线和色卡
# 字符串矩阵
matrix = [
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    ['K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'],
    ['U', 'V', 'W', 'X', 'Y', 'Z', 'A', 'B', 'C', 'D'],
    ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'],
    ['O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'A', 'B'],
    ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']
]

# 创建一个10x10的白色背景图像
canvas = np.ones((10, 10, 3))

# 绘制格子图像
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(canvas)

# 绘制黑色格子线条
for i in range(11):
    plt.plot([-0.5, 9.5], [i - 0.5, i - 0.5], color='black', linewidth=1)
    plt.plot([i - 0.5, i - 0.5], [-0.5, 9.5], color='black', linewidth=1)

# 将字符串填入格子中
for i in range(10):
    for j in range(10):
        text = matrix[i][j]
        plt.text(j, i, text, fontsize=12, va='center', ha='center')

plt.show()
