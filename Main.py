import pandas as pd
import numpy as np

import os
import sys

from Histagram import Hist

print("Введите конечную папку: ")
end_path = input()
print("Введите количество столбцов (-1, если хотите количество столбцов по умолчанию): ")
n = input()


for name in os.listdir("input/"):
    print(name[:-4])
    df = pd.read_csv("input/" + name, encoding="windows-1251", sep=";")
    df = df.drop(df.columns[0], axis=1)

    #os.makedirs("output/" + name[:-4] + "/all", exist_ok=True)
    #os.makedirs("output/" + name[:-4] + "/1", exist_ok=True)
    #os.makedirs("output/" + name[:-4] + "/2", exist_ok=True)
    #os.makedirs("output/" + name[:-4] + "/3", exist_ok=True)
    #os.makedirs("output/" + name[:-4] + "/4", exist_ok=True)

    # df = pd.read_csv("input/" + os.listdir("input/")[0], encoding="windows-1251", sep=";")
    # df = df.drop(df.columns[0], axis=1)

    os.makedirs(end_path + "/" + name[:-4] + "/all", exist_ok=True)
    os.makedirs(end_path + "/" + name[:-4] + "/1", exist_ok=True)
    os.makedirs(end_path + "/" + name[:-4] + "/2", exist_ok=True)
    os.makedirs(end_path + "/" + name[:-4] + "/3", exist_ok=True)
    os.makedirs(end_path + "/" + name[:-4] + "/4", exist_ok=True)

    hist = Hist(name[:-4], np.array(df).T)

    if int(n) == -1:
        hist.save_to_png(path=end_path+"/", noise=True)
    elif int(n) > 0:
        hist.save_to_png(path=end_path + "/", n=int(n), noise=True)

    hist.save_to_csv(path=end_path+"/")
