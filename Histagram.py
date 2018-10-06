import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import scipy as sc
import pandas as pd

from Config import Config


class Hist:

    def __init__(self, name, data):

        self.__name = name
        self.__info = name.split(" ")
        #self.__info = info

        self.__means_list = []
        self.__covars_list = []
        self.__means_minus_list = []
        self.__covars_minus_list = []

        self.__p_values = []

        self.__vertical_1_all = data[0]
        self.__vertical_1_one = data[0][0::4]
        self.__vertical_1_two = data[0][1::4]
        self.__vertical_1_three = data[0][2::4]
        self.__vertical_1_four = data[0][3::4]

        self.__side_1_all = data[1]
        self.__side_1_one = data[1][0::4]
        self.__side_1_two = data[1][1::4]
        self.__side_1_three = data[1][2::4]
        self.__side_1_four = data[1][3::4]

        self.__internal_1_all = data[2]
        self.__internal_1_one = data[2][0::4]
        self.__internal_1_two = data[2][1::4]
        self.__internal_1_three = data[2][2::4]
        self.__internal_1_four = data[2][3::4]

        self.__external_1_all = data[3]
        self.__external_1_one = data[3][0::4]
        self.__external_1_two = data[3][1::4]
        self.__external_1_three = data[3][2::4]
        self.__external_1_four = data[3][3::4]

        self.__vertical_2_all = data[4]
        self.__vertical_2_one = data[4][0::4]
        self.__vertical_2_two = data[4][1::4]
        self.__vertical_2_three = data[4][2::4]
        self.__vertical_2_four = data[4][3::4]

        self.__side_2_all = data[5]
        self.__side_2_one = data[5][0::4]
        self.__side_2_two = data[5][1::4]
        self.__side_2_three = data[5][2::4]
        self.__side_2_four = data[5][3::4]

        self.__internal_2_all = data[6]
        self.__internal_2_one = data[6][0::4]
        self.__internal_2_two = data[6][1::4]
        self.__internal_2_three = data[6][2::4]
        self.__internal_2_four = data[6][3::4]

        self.__external_2_all = data[7]
        self.__external_2_one = data[7][0::4]
        self.__external_2_two = data[7][1::4]
        self.__external_2_three = data[7][2::4]
        self.__external_2_four = data[7][3::4]

        self.__histagram_list = [[self.__vertical_1_all, self.__side_1_all, self.__internal_1_all,
                                  self.__external_1_all,
                                  self.__vertical_2_all, self.__side_2_all, self.__internal_2_all,
                                  self.__external_2_all],

                                 [self.__vertical_1_one, self.__side_1_one, self.__internal_1_one,
                                  self.__external_1_one,
                                  self.__vertical_2_one, self.__side_2_one, self.__internal_2_one,
                                  self.__external_2_one],

                                 [self.__vertical_1_two, self.__side_1_two, self.__internal_1_two,
                                  self.__external_1_two,
                                  self.__vertical_2_two, self.__side_2_two, self.__internal_2_two,
                                  self.__external_2_two],

                                 [self.__vertical_1_three, self.__side_1_three, self.__internal_1_three,
                                  self.__external_1_three,
                                  self.__vertical_2_three, self.__side_2_three, self.__internal_2_three,
                                  self.__external_2_three],

                                 [self.__vertical_1_four, self.__side_1_four, self.__internal_1_four,
                                  self.__external_1_four,
                                  self.__vertical_2_four, self.__side_2_four, self.__internal_2_four,
                                  self.__external_2_four]]

        for item in self.__histagram_list:
                self.__p_values.append([sc.stats.shapiro(item[0]),
                                        sc.stats.shapiro(item[1]),
                                        sc.stats.shapiro(item[2]),
                                        sc.stats.shapiro(item[3]),
                                        sc.stats.shapiro(item[4]),
                                        sc.stats.shapiro(item[5]),
                                        sc.stats.shapiro(item[6]),
                                        sc.stats.shapiro(item[7])])

                GMM = GaussianMixture()
                GMM.fit(item[0].reshape(-1, 1))
                m1 = GMM.means_[0]
                c1 = GMM.covariances_[0]

                plus = item[1][item[1] > 0]
                minus = item[1][item[1] < 0]

                m2 = [-1.00]
                c2 = [[-1.00]]

                if len(plus) != 0:
                    GMM = GaussianMixture()
                    GMM.fit(item[1][item[1] > 0].reshape(-1, 1))
                    m2 = GMM.means_[0]
                    c2 = GMM.covariances_[0]

                m = -1.00
                c = -1.00

                if len(minus) != 0:
                    GMM = GaussianMixture()
                    GMM.fit(item[1][item[1] < 0].reshape(-1, 1))
                    m = GMM.means_[0][0]
                    c = GMM.covariances_[0][0][0]

                GMM = GaussianMixture()
                GMM.fit(item[2].reshape(-1, 1))
                m3 = GMM.means_[0]
                c3 = GMM.covariances_[0]

                GMM = GaussianMixture()
                GMM.fit(item[3].reshape(-1, 1))
                m4 = GMM.means_[0]
                c4 = GMM.covariances_[0]

                GMM = GaussianMixture()
                GMM.fit(item[4].reshape(-1, 1))
                m5 = GMM.means_[0]
                c5 = GMM.covariances_[0]

                plus = item[5][item[5] > 0]
                minus = item[5][item[5] < 0]

                m6 = [-1.00]
                c6 = [[-1.00]]

                if len(plus) != 0:
                    GMM = GaussianMixture()
                    GMM.fit(item[5][item[5] > 0].reshape(-1, 1))
                    m6 = GMM.means_[0]
                    c6 = GMM.covariances_[0]

                mm = -1.00
                cc = -1.00

                if len(minus) != 0:
                    GMM = GaussianMixture()
                    GMM.fit(item[5][item[5] < 0].reshape(-1, 1))
                    mm = GMM.means_[0][0]
                    cc = GMM.covariances_[0][0][0]

                GMM = GaussianMixture()
                GMM.fit(item[6].reshape(-1, 1))
                m7 = GMM.means_[0]
                c7 = GMM.covariances_[0]

                GMM = GaussianMixture()
                GMM.fit(item[7].reshape(-1, 1))
                m8 = GMM.means_[0]
                c8 = GMM.covariances_[0]

                self.__means_list.append([m1[0], m2[0], m3[0], m4[0], m5[0], m6[0], m7[0], m8[0]])
                self.__covars_list.append([c1[0][0], c2[0][0], c3[0][0], c4[0][0], c5[0][0], c6[0][0], c7[0][0], c8[0][0]])

                self.__means_minus_list.append([m, mm])
                self.__covars_minus_list.append([c, cc])

    def save_to_csv(self, path="output/", P=-4.0, noise=False):
        if noise is False:
            for item in zip(self.__histagram_list, Config.names):
                # print(item[1])
                for i in zip(item[0], item[1]):
                    # print(i)
                    pd.DataFrame(i[0].round(3))\
                        .to_csv(path + self.__name + i[1] + ".csv",
                                encoding="utf-8",
                                sep=";",
                                header=["data"])

    def save_to_png(self, path="output/", n=15, P=-4.0, noise=False):
        if noise is False:

            for list_ in zip(self.__histagram_list, Config.names, self.__means_list, self.__covars_list, self.__means_minus_list, self.__covars_minus_list):
                for i in zip(list_[0], list_[1], list_[2], list_[3]):
                    if "боковая" not in i[1]:
                        # print(i)
                        # plt.scatter(i[0])
                        plt.hist(i[0], n)
                        # plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\n$\mu=$  " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()
                    elif "боковая_1" in i[1]:
                        # print(list_[4])
                        # print(i)
                        # plt.scatter(i[0])
                        plt.hist(i[0], n)
                        # plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\nПоложительные $\mu=$ " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)) +
                                  "\nОтрицательные: $\mu=$  " + str(round(list_[4][0], 2)) + "  $\sigma=$ " + str(round(list_[5][0], 2)), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()
                    elif "боковая_2" in i[1]:
                        # print(i)
                        # plt.scatter(i[0])
                        plt.hist(i[0], n)
                        # plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\nПоложительные $\mu=$ " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)) +
                                  "\nОтрицательные: $\mu=$  " + str(round(list_[4][1], 2)) + "  $\sigma=$ " + str(round(list_[5][1], 2)), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()

        elif noise is True:
            for list_ in zip(self.__histagram_list, Config.names, self.__means_list, self.__covars_list, self.__means_minus_list, self.__covars_minus_list):
                for i in zip(list_[0], list_[1], list_[2], list_[3]):
                    # print(i[1])
                    if "боковая" not in i[1]:
                        clear, anomaly = self.__separate(i[0], P)
                        plt.scatter(clear, (-0.2) * np.ones(len(clear)))
                        plt.hist(clear, n)

                        if len(clear) > 3:
                            shapiro = round(sc.stats.shapiro(clear)[0], 2)
                        else:
                            shapiro = None
                        #plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\n$\mu=$  " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)) +
                                  "\np-values= " + str(shapiro), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()
                    elif "боковая_1" in i[1]:
                        clear, anomaly = self.__separate(i[0], P, side=True)
                        #print(clear)
                        plt.scatter(clear, (-0.5) * np.ones(len(clear)))
                        plt.hist(clear, n)
                        #plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\nПоложительные: $\mu=$  " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)) +
                                  "\nОтрицательные: $\mu=$  " + str(round(list_[4][0], 2)) + "  $\sigma=$ " + str(round(list_[5][0], 2)), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()
                    elif "боковая_2" in i[1]:
                        clear, anomaly = self.__separate(i[0], P, side=True)
                        plt.scatter(clear, (-0.5) * np.ones(len(clear)))
                        plt.hist(clear, n)
                        #plt.savefig()
                        plt.title(self.__info[0] + " " +
                                  self.__info[1] + " " +
                                  self.__info[2] + " " +
                                  self.__info[3][:-2].replace(",", ".") + " " +
                                  self.__info[4][:-3] + " " +
                                  self.__info[5] + " " +
                                  self.__info[6] +
                                  "\nПоложительные: $\mu=$  " + str(round(i[2], 2)) + "  $\sigma=$ " + str(round(i[3], 2)) +
                                  "\nОтрицательные: $\mu=$  " + str(round(list_[4][1], 2)) + "  $\sigma=$ " + str(round(list_[5][1], 2)), y=1.05)
                        plt.tight_layout()
                        plt.savefig(path + self.__name + i[1])
                        plt.clf()

    def __separate(self, data, P, side=False):
        if side is False:
            GMM = GaussianMixture()
            GMM.fit(data.reshape(-1, 1))
            scores = np.array(GMM.score_samples(data.reshape(-1, 1)))
            values = data

            scores[scores > P] = True
            scores[scores <= P] = False
            scores = np.array(scores, dtype=bool)

            values_clear = values[scores]
            values_anomaly = values[np.logical_not(scores)]

            return values_clear, values_anomaly

        elif side is True:
            plus = data[data >= 0]
            minus = data[data < 0]

            values_clear_plus = []
            values_clear_minus = []
            values_anomaly_plus = []
            values_anomaly_minus = []

            if len(plus) != 0:
                GMM = GaussianMixture()
                GMM.fit(plus.reshape(-1, 1))
                scores = np.array(GMM.score_samples(plus.reshape(-1, 1)))
                values = plus

                #print(scores)

                scores[scores > P] = True
                scores[scores <= P] = False
                scores = np.array(scores, dtype=bool)

                values_clear_plus = values[scores]
                values_anomaly_plus = values[np.logical_not(scores)]

            if len(minus) != 0:
                GMM = GaussianMixture()
                GMM.fit(minus.reshape(-1, 1))
                scores = np.array(GMM.score_samples(minus.reshape(-1, 1)))
                values = minus

                scores[scores > P] = True
                scores[scores <= P] = False
                scores = np.array(scores, dtype=bool)

                values_clear_minus = values[scores]
                values_anomaly_minus = values[np.logical_not(scores)]

            return list(values_clear_plus) + list(values_clear_minus), list(values_anomaly_plus) + list(values_anomaly_minus)
