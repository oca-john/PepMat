import tracemalloc

tracemalloc.start()
current_memory, peak_memory = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current_memory / 1024 / 1024} MB")
print(f"Peak memory usage: {peak_memory / 1024 / 1024} MB")


def generate_list(data):
    result = [0, 0, 0, 0]  # 初始列表
    for num in data:
        if 1 <= num <= 4:
            result[num - 1] = 1  # 设置对应位置的数据为1
    return result


# 示例数据：(1, 3)
data = (1, 2, 4)
output = generate_list(data)
print(output)

# 获取内存统计信息
current_memory, peak_memory = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current_memory / 1024 / 1024} MB")
print(f"Peak memory usage: {peak_memory / 1024 / 1024} MB")
tracemalloc.stop()
