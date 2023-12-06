import time

# 计算程序运行时间

# 程序执行前
start_time = time.time()  # 获取当前时间

print("Hello BioSensors!")

# 程序执行后
end_time = time.time()  # 获取当前时间
run_time = end_time - start_time  # 计算程序运行时间
run_time_rounded = round(run_time, 3)  # 保留三位小数
print("The program ran for: ", run_time_rounded, "seconds")
