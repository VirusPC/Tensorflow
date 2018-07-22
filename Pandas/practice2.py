import pandas as pd

"""
reindex 方法允许使用未包含在原始 DataFrame 索引值中的索引值。
请试一下，看看如果使用此类值会发生什么！您认为允许此类值的原因是什么？
"""

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(cities)
print('********************************')
print(cities.reindex([2, 1, 5, 0]))

# 这种行为是可取的，因为索引通常是从实际数据中提取的字符串
# 在这种情况下，如果允许出现“丢失的”索引，您将可以轻松使用外部列表重建索引，
# 因为您不必担心会将输入清理掉。
