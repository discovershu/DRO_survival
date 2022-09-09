import pandas as pd
import numpy as np
import torch


#%%
from compute_survival_function import predict_survival_function


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

def seer_preprocess():
    with open('data/seer/SEER.csv', 'r') as f:
        csv_reader = csv.reader(f)
        header = True
        X = []
        y = []
        for row in csv_reader:
            if header:
                header = False
            else:
                age = int(float(row[0])>65) # age>65 represented by 1
                race = int(row[1] == 'White')  # white: 1, non-white: 0

                matital_status = row[2]
                if matital_status == 'Married (including common law)':
                    matital_status = 0
                elif matital_status == 'Divorced':
                    matital_status = 1
                elif matital_status == 'Single (never married)':
                    matital_status = 2
                elif matital_status == 'Widowed':
                    matital_status = 3
                elif matital_status == 'Separated':
                    matital_status = 4

                T_stage = row[4]
                if T_stage == 'T1':
                    T_stage = 0
                elif T_stage == 'T2':
                    T_stage = 1
                elif T_stage == 'T3':
                    T_stage = 2
                elif T_stage == 'T4':
                    T_stage = 3

                N_stage = row[5]
                if N_stage == 'N1':
                    N_stage = 0
                elif N_stage == 'N2':
                    N_stage = 1
                elif N_stage == 'N3':
                    N_stage = 2

                six_stage = row[6]
                if six_stage == 'IIA':
                    six_stage = 0
                elif six_stage == 'IIB':
                    six_stage = 1
                elif six_stage == 'IIIA':
                    six_stage = 2
                elif six_stage == 'IIIB':
                    six_stage = 3
                elif six_stage == 'IIIC':
                    six_stage = 4

                grade = row[7]
                if grade == 'Moderately differentiated; Grade II':
                    grade = 0
                elif grade == 'Poorly differentiated; Grade III':
                    grade = 1
                elif grade == 'Well differentiated; Grade I':
                    grade = 2
                elif grade == 'Undifferentiated; anaplastic; Grade IV':
                    grade = 3

                a_stage = row[8]
                if a_stage == 'Regional':
                    a_stage = 0
                else:
                    a_stage = 1

                tumor_size = float(row[9])

                estrogen_status = row[10]
                if estrogen_status == 'Positive':
                    estrogen_status = 0
                else:
                    estrogen_status = 1

                progesterone_status = row[11]
                if progesterone_status == 'Positive':
                    progesterone_status = 0
                else:
                    progesterone_status = 1

                regional_node_examimed = float(row[12])

                reginol_node_positive = float(row[13])



                d_time = float(row[14])

                death = row[15]
                if death == 'Alive':  #0:Alive, 1:Dead
                    death = bool(0)
                else:
                    death = bool(1)

                X.append([age, race, matital_status, T_stage, N_stage, six_stage, grade,\
                          a_stage, tumor_size, estrogen_status, progesterone_status, regional_node_examimed,\
                          reginol_node_positive])
                y.append([d_time, death])

    g1_data = list()
    g2_data = list()
    g3_data = list()
    g4_data = list()

    g1_event = list()
    g2_event = list()
    g3_event = list()
    g4_event = list()

    g1_time = list()
    g2_time = list()
    g3_time = list()
    g4_time = list()

    for i in range(len(X)):
        if X[i][0]==0 and X[i][1]==0:
            g1_data.append(X[i])
            g1_event.append(y[i][1])
            g1_time.append(y[i][0])
        elif X[i][0]==0 and X[i][1]==1:
            g2_data.append(X[i])
            g2_event.append(y[i][1])
            g2_time.append(y[i][0])
        elif X[i][0]==1 and X[i][1]==0:
            g3_data.append(X[i])
            g3_event.append(y[i][1])
            g3_time.append(y[i][0])
        else:
            g4_data.append(X[i])
            g4_event.append(y[i][1])
            g4_time.append(y[i][0])

    g1_data = np.asarray(g1_data)
    g2_data = np.asarray(g2_data)
    g3_data = np.asarray(g3_data)
    g4_data = np.asarray(g4_data)

    g1_event = np.asarray(g1_event)
    g2_event = np.asarray(g2_event)
    g3_event = np.asarray(g3_event)
    g4_event = np.asarray(g4_event)

    g1_time = np.asarray(g1_time)
    g2_time = np.asarray(g2_time)
    g3_time = np.asarray(g3_time)
    g4_time = np.asarray(g4_time)

    imp_model = SimpleImputer(missing_values=np.nan, strategy='median')

    g1_imputer = imp_model.fit(g1_data)
    g1_data = g1_imputer.transform(g1_data)

    g2_imputer = imp_model.fit(g2_data)
    g2_data = g2_imputer.transform(g2_data)

    g3_imputer = imp_model.fit(g3_data)
    g3_data = g3_imputer.transform(g3_data)

    g4_imputer = imp_model.fit(g4_data)
    g4_data = g4_imputer.transform(g4_data)

    data_x = np.concatenate((g1_data, g2_data, g3_data, g4_data), axis=0)
    data_event = np.concatenate((g1_event, g2_event, g3_event, g4_event), axis=0)
    data_time = np.concatenate((g1_time, g2_time, g3_time, g4_time), axis=0)

    num_columns = ['age', 'race', 'matital_status', 'T_stage', 'N_stage', 'six_stage', 'grade', 'a_stage',\
                   'tumor_size', 'estrogen_status', 'progesterone_status', 'regional_node_examimed',\
                          'reginol_node_positive']

    data_x = pd.DataFrame(data=data_x, columns=num_columns)
    age = data_x['age'].astype(int)
    race = data_x['race'].astype(int)

    protect_attr = (pd.concat([age, race], axis=1)).values

    data_y = np.dtype([('death', data_event.dtype), ('futime', data_time.dtype)])
    data_y = np.empty(len(data_event), dtype=data_y)
    data_y['death'] = data_event.astype(int)
    data_y['futime'] = data_time

    return data_x, data_y, protect_attr

# print(seer_preprocess())





















