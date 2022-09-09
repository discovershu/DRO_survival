import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%% dataset pre-processing

from performance_measures import c_index, brier_score, weighted_c_index, weighted_brier_score,log_partial_lik
from neural_models import negLogLikelihood, linearCoxPH_Regression, MLP, negLogLikelihood_per_sample, negLogLikelihood_per_sample_for_splitting
from fairness_measures import individual_fairness, group_fairness, intersect_fairness, individual_fairness_scale

from sksurv.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc
#%% linear Cox PH model in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
from scipy import optimize
import datetime

parser = argparse.ArgumentParser(description='PyTorch Joint DRO for Survival Analysis')
parser.add_argument('--dataset', type=str, default="SEER", help="choose from 'FLC', 'SUPPORT', 'SEER'")
parser.add_argument('--model', type=str, default="Linear", help="choose from 'Linear' and 'MLP'")
parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--with_scale', type=bool, default=True, help='use scale for testing')
parser.add_argument('--eps', type=float, default=0.15, help='a lower bound on the group proportions e.g. {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}')
parser.add_argument('--split', type=float, default=0.5, help='Splitting training set to two parts. This value is the prop. of second training set for calculating logsumexp')
parser.add_argument('--seed', type=int, default=7, help='Seed')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
out_str = str(args)
print(out_str)


#%%
from compute_survival_function import predict_survival_function  

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

def threshplus(x):
    y = x.copy()
    y[y<0]=0
    return y

def threshplus_tensor(x):
    y = x.clone()
    y[y<0]=0
    return y

def loss_map_chi_factory(loss_values, eps):
    # return lambda x: np.sqrt(2)*(1.0/eps-1.0)*np.sqrt(np.mean(threshplus(loss_values-x)**2.0)) + x
    return lambda x: np.sqrt(2 * ((1.0 / eps - 1.0)** 2.0)+1) * np.sqrt(np.mean(threshplus(loss_values - x) ** 2.0)) + x

def loss_map_chi_factory_tensor(loss_values, eps, opt_eta):
    # return np.sqrt(2)*(1.0/eps-1.0)*torch.sqrt(torch.mean(threshplus_tensor(loss_values-opt_eta)**2.0)) + opt_eta
    return np.sqrt(2 * ((1.0 / eps - 1.0)** 2.0)+1)*torch.sqrt(torch.mean(threshplus_tensor(loss_values-opt_eta)**2.0)) + opt_eta
    
#%% FLC data:
from utilities import prepare_data
from utilities import check_arrays_survival
from flc_data_preprocess import flc_preprocess
from support_data_preprocess import support_preprocess
from compas_data_preprocess import compas_preprocess
from seer_data_preprocess import seer_preprocess

#Survival Data
if args.dataset == 'FLC':
    data_x, data_y, protect_attr = flc_preprocess()
elif args.dataset == 'SUPPORT':
    data_x, data_y, protect_attr = support_preprocess()
elif args.dataset == 'COMPAS':
    data_x, data_y, protect_attr = compas_preprocess()
elif args.dataset == 'SEER':
    data_x, data_y, protect_attr = seer_preprocess()
else:
    print('unknown')

# train-test split
data_X_train, data_X_test, data_y_train, data_y_test, S_train, S_test = train_test_split(data_x, data_y, protect_attr, test_size=0.2,stratify=data_y["death"], random_state=args.seed)
data_X_train, data_X_dev, data_y_train, data_y_dev, S_train, S_dev = train_test_split(data_X_train, data_y_train, S_train, test_size=0.2,stratify=data_y_train["death"], random_state=args.seed)
data_X_train_1, data_X_train_2, data_y_train_1, data_y_train_2, S_train_1, S_train_2 = train_test_split(data_X_train, data_y_train, S_train, test_size=args.split,stratify=data_y_train["death"], random_state=args.seed)
#
data_X_train_1, data_event_train_1, data_time_train_1 = check_arrays_survival(data_X_train_1, data_y_train_1)
data_X_train_2, data_event_train_2, data_time_train_2 = check_arrays_survival(data_X_train_2, data_y_train_2)
data_X_train_1, data_event_train_1, data_time_train_1, S_train_1 = prepare_data(data_X_train_1, data_event_train_1, data_time_train_1, S_train_1)
data_X_train_2, data_event_train_2, data_time_train_2, S_train_2 = prepare_data(data_X_train_2, data_event_train_2, data_time_train_2, S_train_2)

data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_test, data_y_test)
data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test, S_test)
#
intersectionalGroups = np.unique(S_train_1,axis=0) # all intersecting groups, i.e. black-women, white-man etc
# data normalization: mean subtraction method to compute euclidean distance
scaler = StandardScaler()
scaler.fit(data_X_train_1)
data_X_train_1 = scaler.transform(data_X_train_1)
data_X_train_2 = scaler.transform(data_X_train_2)
data_X_test = scaler.transform(data_X_test)

#%%
# hyperparameters of the model
input_size = data_X_train_1.shape[1]
output_size = 1


#%% intialize model and optimizar
# initialize cox PH model   
if args.model == 'Linear':
    coxPH_model = linearCoxPH_Regression(input_size,output_size)
if args.model == 'MLP':
    coxPH_model = MLP(input_size, output_size)
# Loss and optimizer
criterion = negLogLikelihood()
criterion_per_sample = negLogLikelihood_per_sample()
criterion_per_sample_splitting = negLogLikelihood_per_sample_for_splitting()
optimizer = optim.Adam(coxPH_model.parameters(),lr = args.lr) # adam optimizer

#%% training cox ph model 
data_X_train_1 = Variable((torch.from_numpy(data_X_train_1)).float())
data_event_train_1 = Variable((torch.from_numpy(data_event_train_1)).float())
data_time_train_1 = Variable((torch.from_numpy(data_time_train_1)).float())
data_X_train_2 = Variable((torch.from_numpy(data_X_train_2)).float())
data_event_train_2 = Variable((torch.from_numpy(data_event_train_2)).float())
data_time_train_2 = Variable((torch.from_numpy(data_time_train_2)).float())
  
data_X_test = Variable((torch.from_numpy(data_X_test)).float())


starttime = datetime.datetime.now()
#%% stochastic method
for epoch in range(args.epochs):
    #for batch in range(0,np.int64(np.floor(len(data_X_train)/mini_batch))*mini_batch,mini_batch):    
    
    # backward propagation
    outputs_1 = coxPH_model(data_X_train_1)
    outputs_2 = coxPH_model(data_X_train_2)

    per_sample_losses_1 = criterion_per_sample_splitting(outputs_1, data_event_train_1, data_time_train_1, outputs_2, data_time_train_2)  # loss between prediction and target
    chi_loss_np_1 = loss_map_chi_factory(per_sample_losses_1.detach().numpy(), args.eps)
    cutpt_1 = optimize.fminbound(chi_loss_np_1, np.min(per_sample_losses_1.detach().numpy()) - 1000.0, np.max(per_sample_losses_1.detach().numpy()))
    loss_1 = loss_map_chi_factory_tensor(per_sample_losses_1, args.eps, cutpt_1)

    per_sample_losses_2 = criterion_per_sample_splitting(outputs_2, data_event_train_2, data_time_train_2, outputs_1,
                                                         data_time_train_1)  # loss between prediction and target
    chi_loss_np_2 = loss_map_chi_factory(per_sample_losses_2.detach().numpy(), args.eps)
    cutpt_2 = optimize.fminbound(chi_loss_np_2, np.min(per_sample_losses_2.detach().numpy()) - 1000.0,
                                 np.max(per_sample_losses_2.detach().numpy()))
    loss_2 = loss_map_chi_factory_tensor(per_sample_losses_2, args.eps, cutpt_2)

    loss = loss_1 + loss_2


    optimizer.zero_grad() # zero the parameter gradients
    loss.backward()
    optimizer.step()
    # print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()))

endtime = datetime.datetime.now()
time = (endtime - starttime).seconds
print('Training Time:', time)
    
#%% Evaluate the model
# import sys
# sys.stdout=open("typical_Cox_PH_output_Batch.txt","w")
#Measuring the Performance of Survival Models
# linear predictor for train data
with torch.no_grad():
    base_prediction = coxPH_model(data_X_train_1)
    base_prediction = (base_prediction.numpy()).reshape((-1,)) # beta \dot x
# linear predictor for test/dev data
with torch.no_grad():
    model_prediction = coxPH_model(data_X_test)
    model_prediction = (model_prediction.numpy()).reshape((-1,)) # beta \dot x

skSurv_result_test = concordance_index_censored(data_event_test, data_time_test, model_prediction)
print(f"skSurv implemented C-index for test data: {skSurv_result_test[0]: .4f}")
    
# eval_time = [int(np.percentile(data_time_test, 25)), int(np.percentile(data_time_test, 50)), int(np.percentile(data_time_test, 75))]
# tmp_br_score = np.zeros(len(eval_time))
#
data_event_train_1 = data_event_train_1.numpy().astype(bool)
data_time_train_1 = data_time_train_1.numpy()
# #%%
# survFunction_test = predict_survival_function(model_prediction, data_event_test, data_time_test, model_prediction)
# for t in range(len(eval_time)):
#     cif_test = np.zeros((len(data_X_test)))
#     for i in range(len(data_X_test)):
#         time_point = survFunction_test[i].x
#         probs = survFunction_test[i].y
#         index=np.where((time_point==eval_time[t]))[0][0]
#         cif_test[i] = 1 - probs[index]
#
#     tmp_br_score[t] = weighted_brier_score(data_time_train_1, data_event_train_1, cif_test, data_time_test, data_event_test, eval_time[t])
#
# weighted_br_score = np.mean(tmp_br_score)
# print(f"weighted brier score: {weighted_br_score: .4f}")

#%% Time-dependent Area under the ROC
survival_train=np.dtype([('event',data_event_train_1.dtype),('surv_time',data_time_train_1.dtype)])
survival_train=np.empty(len(data_event_train_1),dtype=survival_train)
survival_train['event']=data_event_train_1
survival_train['surv_time']=data_time_train_1

survival_test=np.dtype([('event',data_event_test.dtype),('surv_time',data_time_test.dtype)])
survival_test=np.empty(len(data_event_test),dtype=survival_test)
survival_test['event']=data_event_test
survival_test['surv_time']=data_time_test

event_times = np.arange(np.min(data_time_test), np.max(data_time_test)/2, 75)

test_auc, test_mean_auc = cumulative_dynamic_auc(survival_train, survival_test, model_prediction, event_times)

print(f"Time-dependent Area under the ROC: {test_mean_auc: .4f}")

# plt.plot(event_times, test_auc, marker="o")
# plt.axhline(test_mean_auc, linestyle="--")
# plt.xlabel("Days from Enrollment")
# plt.ylabel("Time-dependent Area under the ROC")
# plt.grid(True)
# plt.savefig('typical_auc_batch.png',dpi = 600)

#%% log -partial likelihood
log_lik = log_partial_lik(model_prediction.reshape(-1,1), data_event_test.reshape(-1,1))
print(f"Log partial likelihood: {log_lik: .4f}")

#%% individual fairness measures
data_X_test_for_distance = data_X_test.numpy()
data_X_test_for_distance = data_X_test_for_distance / np.linalg.norm(data_X_test_for_distance,axis=1,keepdims=1)

if args.with_scale:
    scale_measure = 0.01
    R_beta_scale = individual_fairness_scale(model_prediction, data_X_test_for_distance, scale_measure)
    print(f"average individual fairness metric with scale={scale_measure: .4f}: {R_beta_scale: .4f}")
else:
    R_beta = individual_fairness(model_prediction,data_X_test_for_distance)
    print(f"average individual fairness metric: {R_beta: .4f}")

#%% group fairness measures - age or race
if args.dataset == 'COMPAS':
    S_race = S_test[:, 0]  # race is in the 1st column
    group_fairness_race = group_fairness(model_prediction, S_race)
    print(f"group fairness metric (for race): {group_fairness_race: .4f}")
else:
    S_age = S_test[:,0] # age is in the 1st column
    group_fairness_age = group_fairness(model_prediction,S_age)
    print(f"group fairness metric (for age): {group_fairness_age: .4f}")


#%% intersectional fairness measures
epsilon = intersect_fairness(model_prediction,S_test, intersectionalGroups)
print(f"intersectional fairness metric: {epsilon: .4f}")

#%% save the model
# torch.save(coxPH_model.state_dict(), "trained-models/TypicalCoxPH_model")
