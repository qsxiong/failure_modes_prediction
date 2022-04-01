from scipy.signal import periodogram
import math
import numpy as np
import csv
import matplotlib.pyplot as plt



def interprolate(fLeft,fRight,pLeft,pRight,f):
    '''
    线性插值
    '''
    p = (fRight-f)/(fRight-fLeft)*pLeft + (f-fLeft)/(fRight-fLeft)*pRight
    return p

def powerDistribution(f1, f2, acc, fs):
    '''
    :param f1: 结构已知第一阶基频
    :param f2: 结构已知第二阶频率
    :param acc: 加速度时程,array
    :param fs: 加速度时程采样频率
    :return: 指标
    '''
    # 求功率谱密度
    f, Pxx = periodogram(acc, fs)
    plt.figure()
    plt.plot(f[1:], 10*np.log10(Pxx)[1:])
    # 求前两两阶频率的功率谱密度
    f_space = f[1]-f[0]
    p1 = interprolate(f[int(f1/f_space)],f[int(f1/f_space)+1],Pxx[int(f1/f_space)],Pxx[int(f1/f_space)+1],f1)
    p2 = interprolate(f[int(f2/f_space)],f[int(f2/f_space)+1],Pxx[int(f2/f_space)],Pxx[int(f2/f_space)+1],f2)

    # 二者比值为指标
    indicator = p1 / p2

    return indicator

 # 均值以及标准差计算
def meanAndDev(list):
    '''

    :param list: 前两分钟的indicator的list
    :return: 均值和标准差
    '''

    mean = sum(list) / len(list)
    var = sum((l - mean) ** 2 for l in list) / len(list)
    st_dev = math.sqrt(var)
    return mean, st_dev

def abnormalAlert(list, timeSpace,indicator):
    '''

    :param list: 前两分钟的indicator的list
    :param timeSpace: 计算间隔时间
    :param indicator: 本次计算的指标
    :return: S
    '''
    n = int(2 * 60 / timeSpace)
    if len(list) >= n:
        mean, st_dev = meanAndDev(list[-n:])
        if indicator > mean + 3 * st_dev or indicator < mean - 3 * st_dev:
            print("异常")
    else:
        print("正常")


if __name__ == '__main__':
    f1=0.108
    f2=0.17
    fs = 100
    # 窗口大小
    timespace = 60
    num = timespace*fs
    end=0
    list = []
    # 读取数据
    with open ('./datanew.csv',encoding='gb18030', errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [float(row[1])for row in reader]
        s = np.array(column1)

    while end < s.shape[0]:
        indicator = powerDistribution(f1,f2,s[end:end+num],fs)
        end = end + num
        list.append(indicator)
        abnormalAlert(list, timespace, indicator)





