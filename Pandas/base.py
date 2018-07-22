import pandas as pd
import numpy as np

"""DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。"""

# print(pd.__version__)

# 创建 Series 的一种方法是构建 Series 对象
print(pd.Series(['San Francisco', 'San Jose', 'Sacramento']))

print('********************************************')
# 您可以将映射 string 列名称的 dict 传递到它们各自的 Series，从而创建DataFrame对象。
# 如果 Series 在长度上不一致，系统会用特殊的 NA/NaN 值填充缺失的值。
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
df = pd.DataFrame({'City name': city_names, 'Population': population})
print(df)

# 但是在大多数情况下，您需要将整个文件加载到 DataFrame 中。下面的示例加载了一个包含加利福尼亚州住房数据的文件。
# 请运行以下单元格以加载数据，并创建特征定义：
california_housing_dataframe = pd.read_csv('california_housing_train.csv', sep=',')
print('*****************describe********************')
print(california_housing_dataframe.describe())  # 使用 DataFrame.describe 来显示关于 DataFrame 的有趣统计信息。
print('******************head*********************')
print(california_housing_dataframe.head())  # 显示 DataFrame 的前几个记录, 可传入限制数量参数
print('*******************hist*********************')
california_housing_dataframe.hist('housing_median_age')  # 直方图, 通过matplotlib包操作

