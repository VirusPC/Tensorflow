import pandas as pd

"""
通过添加一个新的布尔值列（当且仅当以下两项均为 True 时为 True）修改 cities 表格：
    城市以圣人命名。
    城市面积大于 50 平方英里。
    
注意：布尔值 Series 是使用“按位”而非传统布尔值“运算符”组合的。例如，执行逻辑与时，应使用 &，而不是 and。

提示："San" 在西班牙语中意为 "saint"。
"""

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['bool'] = (cities['Area square miles'] > 50) & (cities['City name'].apply(lambda name: name.startswith('San')))
print(cities)
