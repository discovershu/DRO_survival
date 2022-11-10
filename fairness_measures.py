import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
#%% fairness metric: individual fairness
def individual_fairness(prediction,X):
    HazardFunction = np.exp(prediction)
# =============================================================================
#     normalizer = np.linalg.norm(model_prediction, 2) # L2 normalization
#     HazardFunction = HazardFunction/normalizer # L2 normalization
# =============================================================================
    N = len(prediction)
    R_beta = 0.0 #initialization of individual fairnessd 
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j<=i:
                continue
            else:
                distance = np.sqrt(sum((X[i]-X[j])**2)) # euclidean distance
                R_beta = R_beta + max(0,(np.abs(HazardFunction[i]-HazardFunction[j])-distance))  
    R_beta_avg = R_beta/(N*(N-1))
    return R_beta_avg

def individual_fairness_scale(prediction,X, scale):
    HazardFunction = np.exp(prediction)
    N = len(prediction)
    R_beta = 0.0 #initialization of individual fairnessd
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j<=i:
                continue
            else:
                distance = np.sqrt(sum((X[i]-X[j])**2)) # euclidean distance
                R_beta = R_beta + max(0,(np.abs(HazardFunction[i]-HazardFunction[j])-scale*distance))
    R_beta_avg = R_beta/(N*(N-1))
    return R_beta_avg


#%% fairness metric: group fairness
def group_fairness(prediction,S): 
    h_ratio = np.exp(prediction)
    unique_group = np.unique(S)
    avg_h_ratio = sum(h_ratio)/len(h_ratio)
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    h_ratio_group = np.zeros((len(unique_group)))
    group_total = np.zeros((len(unique_group)))
    
    for i in range(len(h_ratio)):
        h_ratio_group[S[i]] = h_ratio_group[S[i]] + h_ratio[i]
        group_total[S[i]] = group_total[S[i]] + 1  
    
    avg_h_ratio_group = (h_ratio_group+dirichletAlpha)/(group_total+concentrationParameter)
    
    group_fairness = np.max(np.abs(avg_h_ratio_group-avg_h_ratio))
    return group_fairness
#%% fairness metric: intersectional fairness
def intersect_fairness(prediction,S,intersect_groups):    
    h_ratio = np.exp(prediction)
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    
    h_ratio_group = np.zeros((len(intersect_groups)))
    group_total = np.zeros((len(intersect_groups)))
    
    for i in range(len(h_ratio)):
        index=np.where((intersect_groups==S[i]).all(axis=1))[0][0]
        h_ratio_group[index] = h_ratio_group[index] + h_ratio[i]
        group_total[index] = group_total[index] + 1  
        
    avg_h_ratio_group = (h_ratio_group+dirichletAlpha)/(group_total+concentrationParameter)
    
    epsilon = 0.0 # intersectional fairness
    for i in  range(len(avg_h_ratio_group)):
        for j in range(len(avg_h_ratio_group)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon,abs(np.log(avg_h_ratio_group[i])-np.log(avg_h_ratio_group[j]))) 
    return epsilon

def CI(prediction, data_event, data_time, S):
    C_group = np.zeros(len(np.unique(S)), dtype=float)
    P_group = np.zeros(len(np.unique(S)), dtype=float)

    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j==i:
                continue
            else:
                if ((data_time[i]<data_time[j]) and data_event[i]==False) \
                        or ((data_time[i]>data_time[j]) and data_event[j]==False) \
                        or ((data_time[i]==data_time[j]) and (data_event[i]==False and data_event[j]==False)):
                    continue
                else:
                    P_group[S[i]] = P_group[S[i]] + 1.0
                if data_time[i]<data_time[j]:
                    if prediction[i]>prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif prediction[i]==prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 0.5
                elif data_time[i]>data_time[j]:
                    if prediction[i]<prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif prediction[i]==prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 0.5
                elif data_time[i] == data_time[j]:
                    if data_event[i]==True and data_event[j]==True:
                        if prediction[i]==prediction[j]:
                            C_group[S[i]] = C_group[S[i]] + 1.0
                        else:
                            C_group[S[i]] = C_group[S[i]] + 0.5
                    elif (data_event[i]==False) and (data_event[j]==True) and (prediction[i]<prediction[j]):
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif (data_event[i]==True) and (data_event[j]==False) and (prediction[i]>prediction[j]):
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    else:
                        C_group[S[i]] = C_group[S[i]] + 0.5

    CF = C_group/P_group

    CI = 0
    for i in range(len(np.unique(S))-1):
        for j in range(i+1,len(np.unique(S))):
            temp = np.abs(CF[i] - CF[j])
            if temp > CI:
                CI = temp

    return CI*100

def C_index_difference(unique_elements, S, y_train, event_test, time_test, prediction_test):
    c_index_group = []
    c_td_index_group = []
    for i in unique_elements:
        if len(S.shape)==1:
            data_event_test_group = event_test[S == i]
            data_time_test_group = time_test[S == i]
            model_prediction_group = prediction_test[S == i]
        else:
            data_event_test_group = event_test[(S == i).all(axis=1)]
            data_time_test_group = time_test[(S == i).all(axis=1)]
            model_prediction_group = prediction_test[(S == i).all(axis=1)]
        c_index_group.append(
            concordance_index_censored(data_event_test_group, data_time_test_group, model_prediction_group)[0])
        c_td_index_group.append(concordance_index_ipcw(y_train, Surv.from_arrays(event=data_event_test_group,
                                                                                      time=data_time_test_group),
                                                       model_prediction_group)[0])

    c_index_group_score = 0
    c_td_index_group_score = 0
    for i in range(len(c_index_group) - 1):
        for j in range(i + 1, len(c_index_group)):
            temp1 = np.abs(c_index_group[i] - c_index_group[j])
            temp2 = np.abs(c_td_index_group[i] - c_td_index_group[j])
            if temp1 > c_index_group_score:
                c_index_group_score = temp1
            if temp2 > c_td_index_group_score:
                c_td_index_group_score = temp2
    return c_index_group_score, c_td_index_group_score
