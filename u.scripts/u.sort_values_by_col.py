import pandas as pd

# 创建一个示例数据表格
data = {
    'A': [4, 2, 7, 1, 9, 5, 3, 8, 6, 0, 10, 12, 11, 14, 13, 17, 16, 15, 18,
          19, 20, 23, 22, 21, 25],
    'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'],
    'C': ['foo', 'bar', 'baz', 'qux', 'quux', 'corge', 'grault', 'garply',
          'waldo', 'fred', 'plugh', 'xyzzy', 'thud', 'spam', 'eggs',
          'lobster', 'shrimp', 'crab', 'oyster', 'clam', 'mussel', 'snail',
          'octopus', 'squid', 'jellyfish'],
    'D': [True, False, True, False, True, False, True, False, True, False,
          True, False, True, False, True, False, True, False, True, False,
          True, False, True, False, True]
}

df = pd.DataFrame(data)

# 按照第一列降序排列（有列名的时候需要从数据框取出来指定排序列）
df_sorted = df.sort_values(by=df.columns[0], ascending=False)

print(df_sorted)
