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

def compas_preprocess():
    # with open('data/compas/cox-violent-parsed.csv', 'r') as f:
    with open('data/compas/cox-parsed.csv', 'r') as f:
        csv_reader = csv.reader(f)
        header = True
        filtered_data_1 = []
        for row in csv_reader:
            if header:
                header = False
            else:
                if row[-12]!= "N/A" and (row[9]== 'Caucasian' or row[9]== 'African-American'):
                    id = row[0]
                    race = int(row[9] == 'Caucasian')
                    sex = int(row[5] == 'Male') # Male: 1, Female: 0

                    age_cat = row[8]
                    if age_cat == 'Greater than 45':
                        age_cat = 0
                    elif age_cat == '25 - 45':
                        age_cat = 1
                    elif age_cat == 'Less than 25':
                        age_cat = 2

                    juv_fel_count = float(row[10])
                    juv_misd_count = float(row[12])
                    juv_other_count = float(row[13])
                    priors_count = float(row[14])

                    days_b_screening_arrest = row[15]
                    if days_b_screening_arrest == '':
                        days_b_screening_arrest = np.nan
                    else:
                        days_b_screening_arrest = float(row[15])

                    c_charge_degree = row[22]
                    if 'F' in c_charge_degree:
                        c_charge_degree = 0
                    else:
                        c_charge_degree = 1

                    is_recid = int(row[24])

                    decile_score = float(row[39])

                    score_text = row[-12]
                    if score_text == 'Low':
                        score_text = 0
                    elif score_text == 'Medium':
                        score_text = 1
                    elif score_text == 'High':
                        score_text = 2

                    d_time = float(row[-2]) - float(row[-3])
                    death = bool(row[-1])

                    filtered_data_1.append([id, race, sex, age_cat, juv_fel_count, juv_misd_count, \
                                          juv_other_count, priors_count, days_b_screening_arrest, \
                                          c_charge_degree, is_recid, decile_score, score_text,\
                                          d_time, death])
    filtered_data_2 = []
    for i in range(len(filtered_data_1)):
        if filtered_data_1[i][-2] > 0:
            filtered_data_2.append(filtered_data_1[i])


    X = []
    y = []
    temp_id = -1
    for i in range(len(filtered_data_2)):
        if int(filtered_data_2[i][0]) != temp_id:
            X.append(filtered_data_2[i][1:-2])
            y.append(filtered_data_2[i][-2:])
            temp_id = int(filtered_data_2[i][0])

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

    num_columns = ['race', 'sex', 'age_cat', 'juv_fel_count', 'juv_misd_count', \
                                          'juv_other_count', 'priors_count', 'days_b_screening_arrest', \
                                          'c_charge_degree', 'is_recid', 'decile_score', 'score_text']

    data_x = pd.DataFrame(data=data_x, columns=num_columns)
    race = data_x['race'].astype(int)
    gender = data_x['sex'].astype(int)
    protect_attr = (pd.concat([race, gender], axis=1)).values

    data_y = np.dtype([('death', data_event.dtype), ('futime', data_time.dtype)])
    data_y = np.empty(len(data_event), dtype=data_y)
    data_y['death'] = data_event.astype(int)
    data_y['futime'] = data_time

    return data_x, data_y, protect_attr

# print(compas_preprocess())





















