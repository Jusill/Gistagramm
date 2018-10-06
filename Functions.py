import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from scipy.integrate import quad
import scipy as sc
import math

P = -4.7
num = 12


def all(df, name):
    info = name[:-4].split(" ")

    arr1, arr2 = separate(df[df.columns[0]])

    if len(arr1) != 0:
        plt.hist(df[df.columns[0]], num, color="blue")
    #if len(arr2) != 0:
    #    plt.hist(arr2, num, color="red")

    plt.scatter(arr1, np.zeros(len(arr1)))
    plt.scatter(arr2, np.zeros(len(arr2)), color="red")

    plt.title("Ветикальная, левая   " + "$\mu=$" + ", " + "$\sigma=$" + "")
    plt.savefig("output/" + name[:-4] + "/all/ветикальная_1")
    plt.clf()

    plt.hist(df[df.columns[1]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/боковая_1")
    plt.clf()

    plt.hist(df[df.columns[2]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/внутренняя_1")
    plt.clf()

    plt.hist(df[df.columns[3]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/внешняя_1")
    plt.clf()

    plt.hist(df[df.columns[4]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/ветикальная_2")
    plt.clf()

    plt.hist(df[df.columns[5]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/боковая_2")
    plt.clf()

    plt.hist(df[df.columns[6]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/внутренняя_2")
    plt.clf()

    plt.hist(df[df.columns[7]], num)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/all/внешняя_2")
    plt.clf()

    df[df.columns[0]].round(3).to_csv("output/" + name[:-4] + "/all/вертикальная_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[1]].round(3).to_csv("output/" + name[:-4] + "/all/боковая_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[2]].round(3).to_csv("output/" + name[:-4] + "/all/внутренняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[3]].round(3).to_csv("output/" + name[:-4] + "/all/внешняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[4]].round(3).to_csv("output/" + name[:-4] + "/all/вертикальная_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[5]].round(3).to_csv("output/" + name[:-4] + "/all/боковая_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[6]].round(3).to_csv("output/" + name[:-4] + "/all/внутренняя_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[7]].round(3).to_csv("output/" + name[:-4] + "/all/внешняя_2.csv", sep=";", encoding="utf-8", header=["data"])

    #print(scipy.stats.shapiro(pd.read_csv("output/" + name[:-4] + "/all/боковая_1.csv",
    #                                encoding="windows-1251",
    #                                sep=";"))[1])


def one(df, name):
    plt.hist(df[df.columns[0]][0::4], 25)
    info = name[:-4].split(" ")
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/ветикальная_1")
    plt.clf()

    plt.hist(df[df.columns[1]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/боковая_1")
    plt.clf()

    plt.hist(df[df.columns[2]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/внутренняя_1")
    plt.clf()

    plt.hist(df[df.columns[3]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/внешняя_1")
    plt.clf()

    plt.hist(df[df.columns[4]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/ветикальная_2")
    plt.clf()

    plt.hist(df[df.columns[5]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/боковая_2")
    plt.clf()

    plt.hist(df[df.columns[6]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/внутренняя_2")
    plt.clf()

    plt.hist(df[df.columns[7]][0::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/1/внешняя_2")
    plt.clf()

    df[df.columns[0]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/вертикальная_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[1]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/боковая_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[2]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/внутренняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[3]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/внешняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[4]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/вертикальная_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[5]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/боковая_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[6]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/внутренняя_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[7]][0::4].round(3).to_csv("output/" + name[:-4] + "/1/внешняя_2.csv", sep=";", encoding="utf-8", header=["data"])


def two(df, name):
    plt.hist(df[df.columns[0]][1::4], 25)
    info = name[:-4].split(" ")
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/ветикальная_1")
    plt.clf()

    plt.hist(df[df.columns[1]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/боковая_1")
    plt.clf()

    plt.hist(df[df.columns[2]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/внутренняя_1")
    plt.clf()

    plt.hist(df[df.columns[3]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/внешняя_1")
    plt.clf()

    plt.hist(df[df.columns[4]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/ветикальная_2")
    plt.clf()

    plt.hist(df[df.columns[5]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/боковая_2")
    plt.clf()

    plt.hist(df[df.columns[6]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/внутренняя_2")
    plt.clf()

    plt.hist(df[df.columns[7]][1::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/2/внешняя_2")
    plt.clf()

    df[df.columns[0]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/вертикальная_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[1]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/боковая_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[2]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/внутренняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[3]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/внешняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[4]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/вертикальная_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[5]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/боковая_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[6]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/внутренняя_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[7]][1::4].round(3).to_csv("output/" + name[:-4] + "/2/внешняя_2.csv", sep=";", encoding="utf-8", header=["data"])


def three(df, name):
    plt.hist(df[df.columns[0]][2::4], 25)
    info = name[:-4].split(" ")
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/ветикальная_1")
    plt.clf()

    plt.hist(df[df.columns[1]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/боковая_1")
    plt.clf()

    plt.hist(df[df.columns[2]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/внутренняя_1")
    plt.clf()

    plt.hist(df[df.columns[3]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/внешняя_1")
    plt.clf()

    plt.hist(df[df.columns[4]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/ветикальная_2")
    plt.clf()

    plt.hist(df[df.columns[5]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/боковая_2")
    plt.clf()

    plt.hist(df[df.columns[6]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/внутренняя_2")
    plt.clf()

    plt.hist(df[df.columns[7]][2::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/3/внешняя_2")
    plt.clf()

    df[df.columns[0]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/вертикальная_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[1]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/боковая_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[2]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/внутренняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[3]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/внешняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[4]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/вертикальная_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[5]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/боковая_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[6]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/внутренняя_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[7]][2::4].round(3).to_csv("output/" + name[:-4] + "/3/внешняя_2.csv", sep=";", encoding="utf-8", header=["data"])


def four(df, name):
    plt.hist(df[df.columns[0]][3::4], 25)
    info = name[:-4].split(" ")
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/ветикальная_1")
    plt.clf()

    plt.hist(df[df.columns[1]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/боковая_1")
    plt.clf()

    plt.hist(df[df.columns[2]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/внутренняя_1")
    plt.clf()

    plt.hist(df[df.columns[3]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/внешняя_1")
    plt.clf()

    plt.hist(df[df.columns[4]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/ветикальная_2")
    plt.clf()

    plt.hist(df[df.columns[5]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/боковая_2")
    plt.clf()

    plt.hist(df[df.columns[6]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/внутренняя_2")
    plt.clf()

    plt.hist(df[df.columns[7]][3::4], 25)
    plt.title(info[0] + " " +
              info[1] + " " +
              info[2] + " " +
              info[3][:-2].replace(",", ".") + " " +
              info[4][:-3] + " " +
              info[5] + " " +
              info[6])
    plt.savefig("output/" + name[:-4] + "/4/внешняя_2")
    plt.clf()

    df[df.columns[0]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/вертикальная_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[1]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/боковая_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[2]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/внутренняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[3]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/внешняя_1.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[4]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/вертикальная_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[5]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/боковая_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[6]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/внутренняя_2.csv", sep=";", encoding="utf-8", header=["data"])
    df[df.columns[7]][3::4].round(3).to_csv("output/" + name[:-4] + "/4/внешняя_2.csv", sep=";", encoding="utf-8", header=["data"])


def separate(df):
    GMM = GaussianMixture()
    GMM.fit(df.values.reshape(-1, 1))
    scores = np.array(GMM.score_samples(df.values.reshape(-1, 1)))
    values = df.values

    scores[scores > P] = True
    scores[scores <= P] = False
    scores = np.array(scores, dtype=bool)

    values_clear = values[scores]
    values_anomaly = values[np.logical_not(scores)]

    return values_clear, values_anomaly


def check_norm(name):
    df = pd.read_csv(name, encoding="utf-8", sep=";", engine='python')

    GMM = GaussianMixture()
    GMM.fit(df[df.columns[1]].values.reshape(-1, 1))
    scores = np.array(GMM.score_samples(df[df.columns[1]].values.reshape(-1, 1)))
    values = df[df.columns[1]].values

    scores[scores > P] = True
    scores[scores <= P] = False
    scores = np.array(scores, dtype=bool)

    values = values[scores]

    GMM = GaussianMixture()
    GMM.fit(values.reshape(-1, 1))
    mean = GMM.means_[0]
    mean = mean[0]
    sigma = math.sqrt(GMM.covariances_[0])

    return mean, sigma, sc.stats.shapiro(values)


def check_mix_norm(name):

    mean_1 = None
    mean_2 = None
    sigma_1 = None
    sigma_2 = None
    test_1 = None
    test_2 = None

    df = pd.read_csv(name, encoding="utf-8", sep=";", engine='python')

    X = np.array(df[df.columns[1]].values)

    if len(X[X < 0]) != 0:
        GMM = GaussianMixture()
        GMM.fit(X[X < 0].reshape(-1, 1))
        scores = np.array(GMM.score_samples(X[X < 0].reshape(-1, 1)))
        values = X[X < 0]

        scores[scores > P] = True
        scores[scores <= P] = False
        scores = np.array(scores, dtype=bool)

        #print(np.dstack([scores, values]))

        values = values[scores]

    #print(values)

        GMM = GaussianMixture()
        GMM.fit(values.reshape(-1, 1))
        mean_1 = GMM.means_[0]
        mean_1 = mean_1[0]
        sigma_1 = math.sqrt(GMM.covariances_[0])
        test_1 = sc.stats.shapiro(values)

    if len(X[X > 0]) != 0:
        GMM = GaussianMixture()
        GMM.fit(X[X > 0].reshape(-1, 1))
        scores = np.array(GMM.score_samples(X[X > 0].reshape(-1, 1)))
        values = X[X > 0]

        scores[scores > P] = True
        scores[scores <= P] = False
        scores = np.array(scores, dtype=bool)

        #print(np.dstack([scores, values]))

        values = values[scores]

        #print(values)

        GMM = GaussianMixture()
        GMM.fit(values.reshape(-1, 1))
        mean_2 = GMM.means_[0]
        mean_2 = mean_2[0]
        sigma_2 = math.sqrt(GMM.covariances_[0])
        test_2 = sc.stats.shapiro(values)

    return mean_1, sigma_1, test_1, mean_2, sigma_2, test_2
