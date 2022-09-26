import pandas as pd
import numpy as np
import torch


#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%%
from utilities import prepare_data
from sklearn.impute import SimpleImputer
from utilities import check_arrays_survival
import csv

def support_preprocess():
    with open('data/support/support2.csv', 'r') as f:
        csv_reader = csv.reader(f)
        header = True
        X = []
        y = []
        for row in csv_reader:
            if header:
                header = False
            else:
                row = row[1:]

                age = int(float(row[0])>65) # age>65 represented by 1
                sex = int(row[2] == 'male') # Male: 1, Female: 0
                race = int(row[16] == 'white') # white: 1, non-white: 0

                num_co = int(row[8])
                diabetes = int(row[22])
                dementia = int(row[23])

                ca = row[24]
                if ca == 'no':
                    ca = 0
                elif ca == 'yes':
                    ca = 1
                elif ca == 'metastatic':
                    ca = 2

                meanbp = row[29]
                if meanbp == '':
                    meanbp = np.nan
                else:
                    meanbp = float(meanbp)

                hrt = row[31]
                if hrt == '':
                    hrt = np.nan
                else:
                    hrt = float(hrt)

                resp = row[32]
                if resp == '':
                    resp = np.nan
                else:
                    resp = float(resp)

                temp = row[33]
                if temp == '':
                    temp = np.nan
                else:
                    temp = float(temp)

                wblc = row[30]
                if wblc == '':
                    wblc = np.nan
                else:
                    wblc = float(wblc)

                sod = row[38]
                if sod == '':
                    sod = np.nan
                else:
                    sod = float(sod)

                crea = row[37]
                if crea == '':
                    crea = np.nan
                else:
                    crea = float(crea)

                d_time = float(row[5])
                death = bool(int(row[1]))

                X.append([age, sex, race, num_co, diabetes, dementia, ca,
                          meanbp, hrt, resp, temp, wblc, sod, crea])
                y.append([d_time, death])
    # X = np.array(X)
    # y = np.array(y)

    ###### remove records with nan
    # not_nan_mask = ~np.isnan(X).any(axis=1)
    # X = X[not_nan_mask].tolist()
    # y = y[not_nan_mask].tolist()


    ###### replace nan with median in records
    g1_data = list()
    g2_data = list()
    g3_data = list()
    g4_data = list()
    g5_data = list()
    g6_data = list()
    g7_data = list()
    g8_data = list()

    g1_event = list()
    g2_event = list()
    g3_event = list()
    g4_event = list()
    g5_event = list()
    g6_event = list()
    g7_event = list()
    g8_event = list()

    g1_time = list()
    g2_time = list()
    g3_time = list()
    g4_time = list()
    g5_time = list()
    g6_time = list()
    g7_time = list()
    g8_time = list()

    for i in range(len(X)):
        if X[i][0]==0 and X[i][1]==0 and X[i][2]==0:
            g1_data.append(X[i])
            g1_event.append(y[i][1])
            g1_time.append(y[i][0])
        elif X[i][0]==0 and X[i][1]==0 and X[i][2]==1:
            g2_data.append(X[i])
            g2_event.append(y[i][1])
            g2_time.append(y[i][0])
        elif X[i][0]==0 and X[i][1]==1 and X[i][2]==0:
            g3_data.append(X[i])
            g3_event.append(y[i][1])
            g3_time.append(y[i][0])
        elif X[i][0]==0 and X[i][1]==1 and X[i][2]==1:
            g4_data.append(X[i])
            g4_event.append(y[i][1])
            g4_time.append(y[i][0])
        elif X[i][0]==1 and X[i][1]==0 and X[i][2]==0:
            g5_data.append(X[i])
            g5_event.append(y[i][1])
            g5_time.append(y[i][0])
        elif X[i][0]==1 and X[i][1]==0 and X[i][2]==1:
            g6_data.append(X[i])
            g6_event.append(y[i][1])
            g6_time.append(y[i][0])
        elif X[i][0]==1 and X[i][1]==1 and X[i][2]==0:
            g7_data.append(X[i])
            g7_event.append(y[i][1])
            g7_time.append(y[i][0])
        elif X[i][0]==1 and X[i][1]==1 and X[i][2]==1:
            g8_data.append(X[i])
            g8_event.append(y[i][1])
            g8_time.append(y[i][0])

    g1_data = np.asarray(g1_data)
    g2_data = np.asarray(g2_data)
    g3_data = np.asarray(g3_data)
    g4_data = np.asarray(g4_data)
    g5_data = np.asarray(g5_data)
    g6_data = np.asarray(g6_data)
    g7_data = np.asarray(g7_data)
    g8_data = np.asarray(g8_data)

    g1_event = np.asarray(g1_event)
    g2_event = np.asarray(g2_event)
    g3_event = np.asarray(g3_event)
    g4_event = np.asarray(g4_event)
    g5_event = np.asarray(g5_event)
    g6_event = np.asarray(g6_event)
    g7_event = np.asarray(g7_event)
    g8_event = np.asarray(g8_event)

    g1_time = np.asarray(g1_time)
    g2_time = np.asarray(g2_time)
    g3_time = np.asarray(g3_time)
    g4_time = np.asarray(g4_time)
    g5_time = np.asarray(g5_time)
    g6_time = np.asarray(g6_time)
    g7_time = np.asarray(g7_time)
    g8_time = np.asarray(g8_time)

    imp_model = SimpleImputer(missing_values=np.nan, strategy='median')

    g1_imputer = imp_model.fit(g1_data)
    g1_data = g1_imputer.transform(g1_data)

    g2_imputer = imp_model.fit(g2_data)
    g2_data = g2_imputer.transform(g2_data)

    g3_imputer = imp_model.fit(g3_data)
    g3_data = g3_imputer.transform(g3_data)

    g4_imputer = imp_model.fit(g4_data)
    g4_data = g4_imputer.transform(g4_data)

    g5_imputer = imp_model.fit(g5_data)
    g5_data = g5_imputer.transform(g5_data)

    g6_imputer = imp_model.fit(g6_data)
    g6_data = g6_imputer.transform(g6_data)

    g7_imputer = imp_model.fit(g7_data)
    g7_data = g7_imputer.transform(g7_data)

    g8_imputer = imp_model.fit(g8_data)
    g8_data = g8_imputer.transform(g8_data)

    data_x = np.concatenate((g1_data, g2_data, g3_data, g4_data, g5_data, g6_data, g7_data, g8_data), axis=0)
    data_event = np.concatenate((g1_event, g2_event, g3_event, g4_event, g5_event, g6_event, g7_event, g8_event), axis=0)
    data_time = np.concatenate((g1_time, g2_time, g3_time, g4_time, g5_time, g6_time, g7_time, g8_time), axis=0)

    num_columns = ['age', 'sex', 'race', 'num_co', 'diabetes', 'dementia', 'ca',
     'meanbp', 'hrt', 'resp', 'temp', 'wblc', 'sod', 'crea']

    data_x = pd.DataFrame(data=data_x, columns=num_columns)
    ages = data_x['age'].astype(int)  # age>65 represented by 1
    gender = data_x['sex'].astype(int)
    race = data_x['race'].astype(int)
    protect_attr = (pd.concat([ages, gender, race], axis=1)).values

    data_y = np.dtype([('death', data_event.dtype), ('futime', data_time.dtype)])
    # data_y = np.dtype([('death', bool), ('futime', data_time.dtype)])
    data_y = np.empty(len(data_event), dtype=data_y)
    data_y['death'] = data_event.astype(int)
    data_y['futime'] = data_time

    return data_x, data_y, protect_attr

# print(support_preprocess())
