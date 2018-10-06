import pandas as pd
import numpy as np
import math

import os


def parse_force(file_path):

    excel_file = pd.ExcelFile(file_path)
    df_ = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
    df_out = pd.DataFrame()

    l = list()

    for j in range(8):
        for i in enumerate(list(df_[df_.columns[j]])):
            if i[0] == 0 or i[0] == 1:
                continue
            if math.isnan(float(i[1])) or i[1] == df_[df_.columns[j]]['Среднее']:
                break

            l.append(i[1])

        if j == 0:
            df_out['vertical_force_1'] = np.round(np.array(l), 3)
        elif j == 1:
            df_out['side_force_1'] = np.round(np.array(l), 3)
        elif j == 2:
            df_out['internal_force_1'] = np.round(np.array(l), 3)
        elif j == 3:
            df_out['external_force_1'] = np.round(np.array(l), 3)
        elif j == 4:
            df_out['vertical_force_2'] = np.round(np.array(l), 3)
        elif j == 5:
            df_out['side_force_2'] = np.round(np.array(l), 3)
        elif j == 6:
            df_out['internal_force_2'] = np.round(np.array(l), 3)
        elif j == 7:
            df_out['external_force_2'] = np.round(np.array(l), 3)

        l.clear()

    return df_out


for i1 in os.listdir("КАЧКАНАР"):
    for i2 in os.listdir("КАЧКАНАР/" + i1):
        print(i2)
        parse_force("КАЧКАНАР/" + i1 + "/" + i2).to_csv("output/" + i2[:-5] + ".csv", sep=';')
