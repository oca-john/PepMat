import re

# 用 RE 提取文件名中的编号
# 编号本身包含了类别信息，解析编号的长度和具体数字信息

filename = "gas_134_flat"
pattern = r"gas_([1-4]{1,4})_flat"
match = re.search(pattern, filename)

match_list = list(match.group(1))

print("there are(is)", len(match_list), "kinds of gases.")
print("this is the gases list:", match_list)

for g in match_list:
    print("there is the gas:", g)
