import pandas as pd
import numpy as np

"""Series 和 DataFrame 对象也定义了 index 属性，
该属性会向每个 Series 项或 DataFrame 行赋一个标识符值。
默认情况下，在构造时，pandas 会赋可反映源数据顺序的索引值。
索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。"""

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city_names, 'Population': population})
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

print(cities['City name'].index)  # RangeIndex(start=0, stop=3, step=1)
# print(cities['City name'][0].index)  # RangeIndex(start=0, stop=3, step=1)
# print(city_names[0].index)

# 调用 DataFrame.reindex 以手动重新排列各行的顺序。
# 例如，以下方式与按城市名称排序具有相同的效果：
print(cities)
print(cities.reindex([2, 0, 1]))

# 重建索引是一种随机排列 DataFrame 的绝佳方式。在下面的示例中，我们会取用类似数组的索引，
# 然后将其传递至 NumPy 的 random.permutation 函数，该函数会随机排列其值的位置。如果使用此重新随机排列的数组调用 reindex，
# 会导致 DataFrame 行以同样的方式随机排列。 尝试多次运行以下单元格！
cities.reindex(np.random.permutation(cities.index))
