import pandas as pd

# 可以使用熟悉的 Python dict/list 指令访问 DataFrame 数据
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities))  # <class 'pandas.core.frame.DataFrame'>
print(type(cities['City name']))  # <class 'pandas.core.series.Series'>
print(type(cities['City name'][1]))  # <class 'str'>
print(type(cities[0:2]))  # <class 'pandas.core.frame.DataFrame'>