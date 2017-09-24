# -*- encoding:utf-8 -*-
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

class City_analize():
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

    # Get Value
    def value_dongguan(self):
        data = []
        for row in range(1,self.rows_dongguan):
             data.append(self.dongguan_data.row_values(row))
        data = np.array(data)
        return data

    def value_fushan(self):
        data = []
        for row in range(1,self.rows_fushan):
             data.append(self.fushan_data.row_values(row))
        data = np.array(data)
        return data

    def value_guangzhou(self):
        data = []
        for row in range(1,self.rows_guangzhou):
             data.append(self.guangzhou_data.row_values(row))
        data = np.array(data)
        return data

    def value_shenzhen(self):
        data = []
        for row in range(1,self.rows_shenzhen):
             data.append(self.shenzhen_data.row_values(row))
        data = np.array(data)
        return data

    def value_position_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]),float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_position_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_position_guangzhou(self):
        data = self.value_guangzhou()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_position_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    
    def value_success_position_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]),float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_fail_position_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            if str(data[row][4]) == '0.0':
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_success_position_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_fail_position_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_success_position_guangzhou(self):
        data = self.value_guangzhou()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_fail_position_guangzhou(self):
        data = self.value_guangzhou()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_success_position_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_fail_position_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0:
                data_need.append([float(data[row][1]), float(data[row][2])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_dongguan(self):
        data = self.value_dongguan()
        data_need =[]
        for row in range(len(data)):
            data_need.append([float(data[row][1]),float(data[row][2]),float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_guangzhou(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need
    #  区域总成本
    def value_sum_price_fushan(self):
        data = self.value_fushan()
        sum = 0
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                sum = sum + float(data[row][3])
        return sum


    #  价格高低，地域 与 结果
    def value_price_position_success_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_fail_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_success_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_fail_fushan(self):
        data = self.value_fushan()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_success_guangzhou(self):
        data = self.value_guangzhou()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_fail_guangzhou(self):
        data = self.value_guangzhou()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_success_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 1.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need

    def value_price_position_fail_shenzhen(self):
        data = self.value_shenzhen()
        data_need = []
        for row in range(len(data)):
            if float(data[row][4]) == 0.0:
                data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
        data_need = np.array(data_need)
        return data_need


    # Plot

    #  价格高低 地域 结果 绘图
    def plt_status_positon_price_position_dongguan(self):
        '''
        价格高低 地理位置 3D 图
        :return:
        '''
        df1 = pd.DataFrame(self.value_price_position_success_dongguan(),columns=['x', 'y', 'price'])
        # df2 = pd.DataFrame(self.value_fail_position_dongguan(), columns=['x', 'y'])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df1.x, df1.y, df1.price, cmap='rainbow', c='green')
        # ax.scatter(df2.x, df2.y, df2.price_success, cmap='rainbow', c='green')
        plt.show()

    def plt_densty_price_position_dongguan(self):
        df = self.value_price_position_dongguan()
        df = pd.DataFrame(df,columns=['x', 'y', 'price'])

        df__70 = df[df.price <= 70]
        df_70__ = df[df.price > 70]
        df_70_75 = df_70__[df_70__.price <= 75]
        df_75__ = df_70__[df_70__.price >75]
        df_80_85 = df_75__

        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)

        f, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_aspect("equal")
        ax1 = sns.kdeplot(df.x, df.y,cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)

        # ax1 = sns.kdeplot(df__70.x, df__70.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        # ax1 = sns.kdeplot(df_70_75.x, df_70_75.y, cmap="Reds", shade=False, shade_lowest=False, size=100)
        #
        ax1 = sns.kdeplot(df_80_85.x, df_80_85.y, cmap="Reds", shade=False, shade_lowest=False, size=100)


        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_densty_fail_position_dongguan(self):
        fail_position = self.value_fail_position_dongguan()
        fail_position = pd.DataFrame(fail_position, columns=['x', 'y'])
        total_position = self.value_position_dongguan()
        total_position = pd.DataFrame(total_position, columns=['x','y'])
        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)
        sns.kdeplot(fail_position.x, fail_position.y,cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)
        ax1 = sns.kdeplot(total_position.x, total_position.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_status_positon_price_position_fushan(self):
        '''
        价格高低 地理位置 3D 图
        :return:
        '''
        df1 = pd.DataFrame(self.value_price_position_success_fushan(),columns=['x', 'y', 'price'])

        df2 = pd.DataFrame(self.value_price_position_fail_fushan(), columns=['x', 'y', 'price'])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df1.x, df1.y, df1.price, cmap='rainbow', c='blue')
        ax.scatter(df2.x, df2.y, df2.price, cmap='rainbow', c='red')
        plt.show()

    def plt_densty_price_position_fushan(self):
        df = self.value_price_position_fushan()
        df = pd.DataFrame(df, columns=['x', 'y', 'price'])

        df__70 = df[df.price <= 70]
        df_70__ = df[df.price > 70]
        df_70_75 = df_70__[df_70__.price <= 75]
        df_75__ = df_70__[df_70__.price > 75]
        df_80_85 = df_75__

        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)

        f, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_aspect("equal")
        ax1 = sns.kdeplot(df.x, df.y, cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)

        # ax1 = sns.kdeplot(df__70.x, df__70.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        # ax1 = sns.kdeplot(df_70_75.x, df_70_75.y, cmap="Reds", shade=False, shade_lowest=False, size=100)
        #
        # ax1 = sns.kdeplot(df_80_85.x, df_80_85.y, cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()


        # c = len(City_analize().value_price_position_success_dongguan())

    def plt_densty_fail_position_fushan(self):
        fail_position = self.value_fail_position_fushan()
        fail_position = pd.DataFrame(fail_position, columns=['x', 'y'])
        total_position = self.value_position_fushan()
        total_position = pd.DataFrame(total_position, columns=['x','y'])
        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)
        sns.kdeplot(total_position.x, total_position.y,cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)
        ax1 = sns.kdeplot(fail_position.x, fail_position.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_status_positon_price_position_guangzhou(self):
        '''
        价格高低 地理位置 3D 图
        :return:
        '''
        df1 = pd.DataFrame(self.value_price_position_success_guangzhou(),columns=['x', 'y', 'price'])

        df2 = pd.DataFrame(self.value_price_position_fail_guangzhou(), columns=['x', 'y', 'price'])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df1.x, df1.y, df1.price, cmap='rainbow', c='blue')
        ax.scatter(df2.x, df2.y, df2.price, cmap='rainbow', c='red')
        plt.show()

    def plt_densty_price_position_guangzhou(self):
        df = self.value_price_position_guangzhou()
        df = pd.DataFrame(df, columns=['x', 'y', 'price'])
        print(df)
        df__70 = df[df.price <= 70]
        df_70__ = df[df.price > 70]
        df_70_75 = df_70__[df_70__.price <= 75]
        df_75__ = df_70__[df_70__.price > 75]
        df_80_85 = df_75__

        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)

        f, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_aspect("equal")
        ax1 = sns.kdeplot(df.x, df.y, cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)

        ax1 = sns.kdeplot(df__70.x, df__70.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        # ax1 = sns.kdeplot(df_70_75.x, df_70_75.y, cmap="Reds", shade=False, shade_lowest=False, size=100)
        #
        # ax1 = sns.kdeplot(df_80_85.x, df_80_85.y, cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_densty_fail_position_guangzhou(self):
        fail_position = self.value_fail_position_guangzhou()
        fail_position = pd.DataFrame(fail_position, columns=['x', 'y'])
        total_position = self.value_position_guangzhou()
        total_position = pd.DataFrame(total_position, columns=['x','y'])
        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)
        sns.kdeplot(total_position.x, total_position.y,cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)
        ax1 = sns.kdeplot(fail_position.x, fail_position.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_status_positon_price_position_shenzhen(self):
        '''
        价格高低 地理位置 3D 图
        :return:
        '''
        df1 = pd.DataFrame(self.value_price_position_success_shenzhen(),columns=['x', 'y', 'price'])

        df2 = pd.DataFrame(self.value_price_position_fail_shenzhen(), columns=['x', 'y', 'price'])
        print(df1)
        print(df2)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df1.x, df1.y, df1.price, cmap='rainbow', c='blue')
        ax.scatter(df2.x, df2.y, df2.price, cmap='rainbow', c='red')
        plt.show()

    def plt_densty_price_position_shenzhen(self):
        df = self.value_price_position_shenzhen()
        df = pd.DataFrame(df, columns=['x', 'y', 'price'])
        print(df)
        df__70 = df[df.price <= 70]
        df_70__ = df[df.price > 70]
        df_70_75 = df_70__[df_70__.price <= 75]
        df_75__ = df_70__[df_70__.price > 75]
        df_80_85 = df_75__

        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)

        f, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_aspect("equal")
        ax1 = sns.kdeplot(df.x, df.y, cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)

        # ax1 = sns.kdeplot(df__70.x, df__70.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        # ax1 = sns.kdeplot(df_70_75.x, df_70_75.y, cmap="Reds", shade=False, shade_lowest=False, size=100)
        #
        ax1 = sns.kdeplot(df_80_85.x, df_80_85.y, cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_densty_fail_position_shenzhen(self):
        fail_position = self.value_fail_position_shenzhen()
        fail_position = pd.DataFrame(fail_position, columns=['x', 'y'])
        total_position = self.value_position_shenzhen()
        total_position = pd.DataFrame(total_position, columns=['x','y'])
        sns.set(style="white", color_codes=True)
        sns.set(style="darkgrid")
        sns.set(color_codes=True)
        sns.kdeplot(total_position.x, total_position.y,cmap="Blues", shade=True, shade_lowest=False, cbar=False, gridsize=50, size=50)
        ax1 = sns.kdeplot(fail_position.x, fail_position.y,cmap="Reds", shade=False, shade_lowest=False, size=100)

        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]
        plt.show()

    def plt_3D_price_position_dongguan(self):
        df = self.value_price_position_shenzhen()
        df = pd.DataFrame(df, columns=['x', 'y', 'price'])
        df.sort_values(by='x',ascending=True)
        df.sort_values(by='y',ascending=True)
        df.x, df.y, df.price,
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.bar(df.x, df.y, df.price, zdir='y', color='r', alpha=0.8)
        ax.plot_trisurf(df.x, df.y, df.price, linewidth=0.2, antialiased=True)
        plt.show()

    # def plt_densty_fail_position_price_fushan(self):


    def plt_cluster_position_price_finish_fushan(self):
        self.plt_status_positon_price_position_fushan()  # 价格，位置分布3d图
        self.plt_densty_price_position_fushan()         # 分类募集度图
        # 通过图像判定区域的划分数量
        data = self.value_price_position_fushan()
        mission_suuess = len(self.value_success_position_fushan())
        print(mission_suuess)  #成功的任务数量
        clf = KMeans(n_clusters=4)
        # X = self.value_position_fushan()
        X = data

        y_pred = clf.fit_predict(X)

        # 坐标
        x1 = []
        y1 = []

        x2 = []
        y2 = []

        x3 = []
        y3 = []

        x4 =[]
        y4 = []
        # 分布获取类标为0、1、2的数据 赋值给(x1,y1) (x2,y2) (x3,y3)
        i = 0
        while i < len(X):
            if y_pred[i] == 0:
                x1.append(X[i][0])
                y1.append(X[i][1])
            elif y_pred[i] == 1:
                x2.append(X[i][0])
                y2.append(X[i][1])
            elif y_pred[i] == 2:
                x3.append(X[i][0])
                y3.append(X[i][1])
            elif y_pred[i] == 3:
                x4.append(X[i][0])
                y4.append(X[i][1])
            i = i + 1

        plot1, = plt.plot(x1, y1, 'or', marker="x")
        plot2, = plt.plot(x2, y2, 'og', marker="o")
        plot3, = plt.plot(x3, y3, 'ob', marker="*")
        plot4, = plt.plot(x3, y3, 'ob', marker="+")

        # 绘制标题
        plt.title("Kmeans-Basketball Data")

        # 绘制x轴和y轴坐标
        plt.xlabel("assists_per_minute")
        plt.ylabel("points_per_minute")

        # 设置右上角图例
        plt.legend((plot1, plot2, plot3, plot4), ('A', 'B', 'C', 'D'), fontsize=10)

        plt.show()
        # fail_position = self.value_fail_position_fushan()
        # price = self.value_price_position_fushan()






# print(c)
# t = City_analize().plt_densty_fail_position_shenzhen()
# City_analize().plt_status_positon_price_position_fushan()
# print(t)
# print(City_analize().value_sum_price_fushan())
# print(len(City_analize().value_fushan()))

