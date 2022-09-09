import numpy as np
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