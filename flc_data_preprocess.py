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
from sksurv.datasets import load_flchain

def flc_preprocess():
    #Survival Data
    data_x, data_y = load_flchain()
    
    num_columns = ['age','creatinine','flc.grp', 'kappa','lambda','sex']
    data_x = data_x.loc[:, num_columns]
    
    data_x['sex'] = (data_x['sex'] == 'M').astype(int) # Male: 1, Female: 0
    gender =  data_x['sex'].astype(int) # Male: 1, Female: 0
    ages = (data_x['age']>65).astype(int) # age>65 represented by 1
    #data_x['mgus'] = (data_x['mgus'] == 'yes').astype(int) 
    
    data_event = data_y["death"]
    data_time = data_y["futime"]
    
    data_x = data_x.values
    
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
    
    for i in range(len(data_x)):
        if gender[i]==0 and ages[i]==0:
            g1_data.append(data_x[i])
            g1_event.append(data_event[i])
            g1_time.append(data_time[i])
        elif gender[i]==0 and ages[i]==1:
            g2_data.append(data_x[i])
            g2_event.append(data_event[i])
            g2_time.append(data_time[i])
        elif gender[i]==1 and ages[i]==0:
            g3_data.append(data_x[i])
            g3_event.append(data_event[i])
            g3_time.append(data_time[i])
        else:
            g4_data.append(data_x[i])
            g4_event.append(data_event[i])
            g4_time.append(data_time[i])
    
    g1_data=np.asarray(g1_data)
    g2_data=np.asarray(g2_data)
    g3_data=np.asarray(g3_data)
    g4_data=np.asarray(g4_data)
    
    g1_event=np.asarray(g1_event)
    g2_event=np.asarray(g2_event)
    g3_event=np.asarray(g3_event)
    g4_event=np.asarray(g4_event)
    
    g1_time=np.asarray(g1_time)
    g2_time=np.asarray(g2_time)
    g3_time=np.asarray(g3_time)
    g4_time=np.asarray(g4_time)
    
    
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
    
    data_x = pd.DataFrame(data=data_x, columns=num_columns)
    ages = (data_x['age']>65).astype(int) # age>65 represented by 1
    gender = data_x['sex'].astype(int)
    protect_attr = (pd.concat([ages,gender],axis=1)).values
    
    data_y=np.dtype([('death',data_event.dtype),('futime',data_time.dtype)])
    data_y=np.empty(len(data_event),dtype=data_y)
    data_y['death']=data_event
    data_y['futime']=data_time
    return data_x, data_y, protect_attr
    
# print(flc_preprocess())