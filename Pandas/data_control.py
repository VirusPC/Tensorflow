import pandas as pd
import numpy as np

# 可以向Series应用Python的基本运算指令。
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(population/1000)

# NumPy是一种用于进行可续计算的常用工具包。pandas Series 可用作大多数 NumPy 函数的参数：
print(np.log(population))

# 对于更复杂的单列转换，您可以使用 Series.apply。像 Python 映射函数一样，
# Series.apply 将以参数形式接受 lambda 函数，而该函数会应用于每个值。
# 下面的示例创建了一个指明population是否筹够100万的新Series：
print("*********************apply*************************")
print(population.apply(lambda val: val > 1000000))

# DataFrames 的修改方式也非常简单。例如，以下代码向现有 DataFrame 添加了两个 Series：
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities
