'''

2.1 基本思想

划分聚类算法是根据给定的n 个对象或者元组的数据集，构建k 个划分聚类的方法。每个划分即为一个聚簇，并且k  n。该方法将数据划分为k 个组，每个组至少有一个对象，每个对象必须属于而且只能属于一个组。1该方法的划分采用按照给定的k 个划分要求，先给出一个初始的划分，然后用迭代重定位技术，通过对象在划分之间的移动来改进划分。

为达到划分的全局最优，划分的聚类可能会穷举所有可能的划分。但在实际操作中，往往采用比较流行的k-means 算法或者k-median 算法。

2.2 算法步骤

k-means 算法最为简单，实现比较容易。每个簇都是使用对象的平均值来表示。

步骤一：将所有对象随机分配到k 个非空的簇中。

步骤二：计算每个簇的平均值，并用该平均值代表相应的值。

步骤三：根据每个对象与各个簇中心的距离，分配给最近的簇。

步骤四：转到步骤二，重新计算每个簇的平均值。这个过程不断重复直到满足某个准则函数或者终止条件。终止(收敛)条件可以是以下任何一个：没有(或者最小数目)数据点被重新分配给不同的聚类;没有(或者最小数目)聚类中心再发生变化;误差平方和(SSE)局部最小。
'''

from math import pi, sin, cos
from collections import namedtuple
from random import random, choice
from copy import copy
import random
import xlrd
import xlrd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from mpl_toolkits.mplot3d import Axes3D
import re
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#
from numpy import *

from sklearn.cluster import KMeans
from matplotlib.mlab import griddata
import matplotlib.cbook as cbook

try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100

dongguan_data_excel = xlrd.open_workbook('dongguan.xlsx')
fushan_data_exccel = xlrd.open_workbook('fushan.xlsx')
guangzhou_data_excel = xlrd.open_workbook('guangzhou.xlsx')
shenzhen_data_excel = xlrd.open_workbook('shenzhen.xlsx')

dongguan_data = dongguan_data_excel.sheets()[0]
fushan_data = fushan_data_exccel.sheets()[0]
guangzhou_data = guangzhou_data_excel.sheets()[0]
shenzhen_data = shenzhen_data_excel.sheets()[0]

rows_dongguan = dongguan_data.nrows
rows_fushan = fushan_data.nrows
rows_guangzhou = guangzhou_data.nrows
rows_shenzhen = shenzhen_data.nrows


def value_dongguan():
    data = []
    for row in range(1, rows_dongguan):
        data.append(dongguan_data.row_values(row))
    data = np.array(data)
    return data


def value_fushan():
    data = []
    for row in range(1, rows_fushan):
        data.append(fushan_data.row_values(row))
    data = np.array(data)
    return data


def value_guangzhou():
    data = []
    for row in range(1, rows_guangzhou):
        data.append(guangzhou_data.row_values(row))
    data = np.array(data)
    return data


def value_shenzhen():
    data = []
    for row in range(1, rows_shenzhen):
        data.append(shenzhen_data.row_values(row))
    data = np.array(data)
    return data

def value_position_dongguan():
    data = value_dongguan()
    data_need = []
    for row in range(len(data)):
        data_need.append([float(data[row][1]), float(data[row][2])])
    data_need = np.array(data_need)
    return data_need


def value_position_fushan():
    data = value_fushan()
    data_need = []
    for row in range(len(data)):
        data_need.append([float(data[row][1]), float(data[row][2])])
    data_need = np.array(data_need)
    return data_need


def value_position_guangzhou():
    data = value_guangzhou()
    data_need = []
    for row in range(len(data)):
        data_need.append([float(data[row][1]), float(data[row][2])])
    data_need = np.array(data_need)
    return data_need


def value_position_shenzhen():
    data = value_shenzhen()
    data_need = []
    for row in range(len(data)):
        data_need.append([float(data[row][1]), float(data[row][2])])
    data_need = np.array(data_need)
    return data_need

class Point:
    __slots__ = ["x", "y", "group"]

    def __init__(self, x=0.0, y=0.0, group=0):
        self.x, self.y, self.group = x, y, group


def generate_points(npoints, radius):
    points = [Point() for _ in range(npoints)]

    # note: this is not a uniform 2-d distribution
    for p in points:
        r = random.random() * radius
        ang = random.random() * 2 * pi
        p.x = r * cos(ang)
        p.y = r * sin(ang)

    return points


def nearest_cluster_center(point, cluster_centers):
    """Distance and index of the closest cluster center"""

    def sqr_distance_2D(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

    min_index = point.group
    min_dist = FLOAT_MAX

    for i, cc in enumerate(cluster_centers):
        d = sqr_distance_2D(cc, point)
        if min_dist > d:
            min_dist = d
            min_index = i

    return (min_index, min_dist)


def kpp(points, cluster_centers):
    cluster_centers[0] = copy(choice(points))
    d = [0.0 for _ in range(len(points))]

    for i in range(1, len(cluster_centers)):
        sum = 0
        for j, p in enumerate(points):
            d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
            sum += d[j]

        sum *= random()

        for j, di in enumerate(d):
            sum -= di
            if sum > 0:
                continue
            cluster_centers[i] = copy(points[j])
            break

    for p in points:
        p.group = nearest_cluster_center(p, cluster_centers)[0]


def lloyd(points, nclusters):
    cluster_centers = [Point() for _ in range(nclusters)]

    # call k++ init
    kpp(points, cluster_centers)

    lenpts10 = len(points) >> 10

    changed = 0
    while True:
        # group element for centroids are used as counters
        for cc in cluster_centers:
            cc.x = 0
            cc.y = 0
            cc.group = 0

        for p in points:
            cluster_centers[p.group].group += 1
            cluster_centers[p.group].x += p.x
            cluster_centers[p.group].y += p.y

        for cc in cluster_centers:
            cc.x /= cc.group
            cc.y /= cc.group

        # find closest centroid of each PointPtr
        changed = 0
        for p in points:
            min_i = nearest_cluster_center(p, cluster_centers)[0]
            if min_i != p.group:
                changed += 1
                p.group = min_i

        # stop when 99.9% of points are good
        if changed <= lenpts10:
            break

    for i, cc in enumerate(cluster_centers):
        cc.group = i

    return cluster_centers


def print_eps(points, cluster_centers, W=400, H=400):
    Color = namedtuple("Color", "r g b");

    colors = []
    for i in range(len(cluster_centers)):
        colors.append(Color((3 * (i + 1) % 11) / 11.0,
                            (7 * i % 11) / 11.0,
                            (9 * i % 11) / 11.0))

    max_x = max_y = -FLOAT_MAX
    min_x = min_y = FLOAT_MAX

    for p in points:
        if max_x < p.x: max_x = p.x
        if min_x > p.x: min_x = p.x
        if max_y < p.y: max_y = p.y
        if min_y > p.y: min_y = p.y

    scale = min(W / (max_x - min_x),
                H / (max_y - min_y))
    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2

    print("%%!PS-Adobe-3.0\n%%%%BoundingBox: -5 -5 %d %d" % (W + 10, H + 10))

    print("/l {rlineto} def /m {rmoveto} def\n" +
          "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n" +
          "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath " +
          "   gsave 1 setgray fill grestore gsave 3 setlinewidth" +
          " 1 setgray stroke grestore 0 setgray stroke }def")

    for i, cc in enumerate(cluster_centers):
        print("%g %g %g setrgbcolor" %
              (colors[i].r, colors[i].g, colors[i].b))

        for p in points:
            if p.group != i:
                continue
            print("%.3f %.3f c" % ((p.x - cx) * scale + W / 2,
                                   (p.y - cy) * scale + H / 2))

        print("\n0 setgray %g %g s" % ((cc.x - cx) * scale + W / 2,
                                       (cc.y - cy) * scale + H / 2))

    print
    "\n%%%%EOF"


def main():
    npoints = 30000
    k = 7  # # clusters
    points = generate_points(npoints, 10)
    print(points)
    # cluster_centers = lloyd(value_position_fushan(), k)
    # print_eps(value_position_fushan(), cluster_centers)


main()