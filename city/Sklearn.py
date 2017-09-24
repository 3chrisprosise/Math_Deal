from sklearn.cluster import Birch
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果，20行数据，每个y_pred对应X一行或一个球员，聚成3类，类标为0、1、2
print(y_pred)

""" 
第四部分：可视化绘图 
Python导入Matplotlib包，专门用于绘图 
import matplotlib.pyplot as plt 此处as相当于重命名，plt用于显示图像 
"""

import numpy as np
import matplotlib.pyplot as plt

# 获取第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(y)

y = [n[1] for n in X]
print(x)

# 绘制散点图 参数：x横轴 y纵轴 c=y_pred聚类预测结果 marker类型 o表示圆点 *表示星型 x表示点
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()