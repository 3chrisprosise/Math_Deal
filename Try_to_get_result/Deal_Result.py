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

from matplotlib.mlab import griddata
import matplotlib.cbook as cbook


class City_analize():
    dongguan_data_excel = xlrd.open_workbook('dongguan.xlsx')
    fushan_data_exccel = xlrd.open_workbook('fushan.xlsx')
    guangzhou_data_excel = xlrd.open_workbook('guangzhou.xlsx')
    shenzhen_data_excel = xlrd.open_workbook('shenzhen.xlsx')

    person_dongguan_excel = xlrd.open_workbook('person_dongguang.xlsx')
    person_fushan_excel = xlrd.open_workbook('person_foshan.xlsx')
    person_guangzhou_excel = xlrd.open_workbook('person_guangzhou.xlsx')
    person_shenzhen_excel = xlrd.open_workbook('person_shenzhen.xlsx')

    dongguan_data = dongguan_data_excel.sheets()[0]
    fushan_data = fushan_data_exccel.sheets()[0]
    guangzhou_data = guangzhou_data_excel.sheets()[0]
    shenzhen_data = shenzhen_data_excel.sheets()[0]

    person_dongguan_data = person_dongguan_excel.sheets()[0]
    person_fushan_data = person_dongguan_excel.sheets()[0]
    person_guangzhou_data = person_dongguan_excel.sheets()[0]
    person_shenzhen_data = person_dongguan_excel.sheets()[0]


    rows_dongguan = dongguan_data.nrows
    rows_fushan = fushan_data.nrows
    rows_guangzhou = guangzhou_data.nrows
    rows_shenzhen = shenzhen_data.nrows

    rows_person_dongguan = person_dongguan_data.nrows
    rows_person_fushan = person_fushan_data.nrows
    rows_person_guangzhou = person_guangzhou_data.nrows
    rows_person_shenzhen = person_shenzhen_data.nrows

    # Get Mission Value
    def value_dongguan(self):
        data = []
        for row in range(1, self.rows_dongguan):
            data.append(self.dongguan_data.row_values(row))
        data = np.array(data)
        return data

    def value_fushan(self):
        data = []
        for row in range(1, self.rows_fushan):
            data.append(self.fushan_data.row_values(row))
        data = np.array(data)
        return data

    def value_guangzhou(self):
        data = []
        for row in range(1, self.rows_guangzhou):
            data.append(self.guangzhou_data.row_values(row))
        data = np.array(data)
        return data

    def value_shenzhen(self):
        data = []
        for row in range(1, self.rows_shenzhen):
            data.append(self.shenzhen_data.row_values(row))
        data = np.array(data)
        return data

    def value_position_dongguan(self):
        data = self.value_dongguan()
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2])])
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
                data_need.append([float(data[row][1]), float(data[row][2])])
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
        data_need = []
        for row in range(len(data)):
            data_need.append([float(data[row][1]), float(data[row][2]), float(data[row][3])])
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



    # Get Person_Value

if __name__=='__main__':
    Z = 0  # 最大可完成任务数量，先计算一个行政区的任务数量
