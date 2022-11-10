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
from fairness_measures import individual_fairness, group_fairness, intersect_fairness, individual_fairness_scale, CI, C_index_difference

from sksurv.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc
from sksurv.util import Surv
import os
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
parser.add_argument('--protect_index', type=int, default=0, help='protect attribute index')
parser.add_argument('--train_or_evaluation', type=int, default=0, help='0 for training, 1 for evaluation')


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
data_X_train, data_event_train, data_time_train = check_arrays_survival(data_X_train, data_y_train)
data_X_train, data_event_train, data_time_train, S_train = prepare_data(data_X_train, data_event_train, data_time_train, S_train)
data_X_train_1, data_event_train_1, data_time_train_1 = check_arrays_survival(data_X_train_1, data_y_train_1)
data_X_train_2, data_event_train_2, data_time_train_2 = check_arrays_survival(data_X_train_2, data_y_train_2)
data_X_train_1, data_event_train_1, data_time_train_1, S_train_1 = prepare_data(data_X_train_1, data_event_train_1, data_time_train_1, S_train_1)
data_X_train_2, data_event_train_2, data_time_train_2, S_train_2 = prepare_data(data_X_train_2, data_event_train_2, data_time_train_2, S_train_2)

if args.train_or_evaluation==0:
    data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_dev, data_y_dev)
    data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test, S_dev)
    data_y_test = data_y_dev
else:
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
data_X_train = Variable((torch.from_numpy(data_X_train)).float())
data_event_train = Variable((torch.from_numpy(data_event_train)).float())
data_time_train = Variable((torch.from_numpy(data_time_train)).float())
data_X_train_1 = Variable((torch.from_numpy(data_X_train_1)).float())
data_event_train_1 = Variable((torch.from_numpy(data_event_train_1)).float())
data_time_train_1 = Variable((torch.from_numpy(data_time_train_1)).float())
data_X_train_2 = Variable((torch.from_numpy(data_X_train_2)).float())
data_event_train_2 = Variable((torch.from_numpy(data_event_train_2)).float())
data_time_train_2 = Variable((torch.from_numpy(data_time_train_2)).float())
  
data_X_test = Variable((torch.from_numpy(data_X_test)).float())

saved_model_name = "saved_models/DRO_COX_SPLIT_{}_{}_lr_{}_seed_{}_sensitive_{}_alpha_{}_split_{}"\
    .format(args.dataset, args.model, args.lr, args.seed, args.protect_index, args.eps, args.split)
if args.train_or_evaluation==0:
    print('Model Training')
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

    # %% save the model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(coxPH_model.state_dict(), saved_model_name)
else:
    print('Load the Saved Model')
    coxPH_model.load_state_dict(torch.load(saved_model_name))
    
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

print('----------------------Accuracy Measures----------------------')

skSurv_result_test = concordance_index_censored(data_event_test, data_time_test, model_prediction)
print(f"skSurv implemented C-index for test data: {skSurv_result_test[0]: .4f}")



# skSurv_C_td_result_test = concordance_index_ipcw(Surv.from_arrays(event=data_event_train, time=data_time_train),
#                                                  Surv.from_arrays(event=data_event_test, time=data_time_test),
#                                                  model_prediction)
# print(f"skSurv implemented C^td-index for test data: {skSurv_C_td_result_test[0]: .4f}")

data_event_train = data_event_train.numpy().astype(bool)
data_time_train = data_time_train.numpy()
survFunction_test = predict_survival_function(base_prediction, data_event_train, data_time_train, model_prediction)

#####brier score
# eval_time = [int(np.percentile(data_time_train, 25)), int(np.percentile(data_time_train, 50)), int(np.percentile(data_time_train, 75))]
# tmp_br_score = np.zeros(len(eval_time))
# tmp_br_score_from_sksurv = np.zeros(len(eval_time))
#
# for t in range(len(eval_time)):
#     cif_test = np.zeros((len(data_X_test)))
#     for i in range(len(data_X_test)):
#         time_point = survFunction_test[i].x
#         probs = survFunction_test[i].y
#         index=np.where((time_point==eval_time[t]))[0][0]
#         cif_test[i] = 1 - probs[index]
#
#     tmp_br_score[t] = weighted_brier_score(data_time_train, data_event_train, cif_test, data_time_test, data_event_test, eval_time[t])
#
# weighted_br_score = np.mean(tmp_br_score)
# print(f"weighted brier score: {weighted_br_score: .4f}")
#
# for t in range(len(eval_time)):
#     preds_bs = [fn(eval_time[t]) for fn in survFunction_test]
#     _, tmp_br_score_from_sksurv[t] = brier_score(data_y_train, data_y_test, preds_bs, eval_time[t])
# print(f"weighted brier score from sksurv: {np.mean(tmp_br_score_from_sksurv): .4f}")


#####integrated brier score

percentiles = [100]
tmp_IBS_from_sksurv = np.zeros(len(percentiles))
min_time = 0
if data_time_train.min()>data_time_test.min():
    min_time = data_time_train.min()
else:
    min_time = data_time_test.min()
for t in range(len(percentiles)):
    times_range = np.linspace(min_time, np.percentile(data_time_test, percentiles[t]), num=50, endpoint=False)
    preds_ibs = np.asarray([[fn(i_t) for i_t in times_range] for fn in survFunction_test])
    tmp_IBS_from_sksurv[t] = integrated_brier_score(data_y_train, data_y_test, preds_ibs, times_range)
    print("integrated brier score from sksurv (percentile {}): {:0.4f}".format(percentiles[t], tmp_IBS_from_sksurv[t]))



# %% Time-dependent Area under the ROC
survival_train = np.dtype([('event', data_event_train.dtype), ('surv_time', data_time_train.dtype)])
survival_train = np.empty(len(data_event_train), dtype=survival_train)
survival_train['event'] = data_event_train
survival_train['surv_time'] = data_time_train

survival_test = np.dtype([('event', data_event_test.dtype), ('surv_time', data_time_test.dtype)])
survival_test = np.empty(len(data_event_test), dtype=survival_test)
survival_test['event'] = data_event_test
survival_test['surv_time'] = data_time_test

# event_times = np.arange(np.min(data_time_test), np.max(data_time_test) / 2, 75)
event_times = np.percentile(data_time_test, np.linspace(5, 81, 15))
# event_times = np.percentile(data_y['futime'], np.linspace(5, 81, 15))

test_auc, test_mean_auc = cumulative_dynamic_auc(survival_train, survival_test, model_prediction, event_times)

print(f"Time-dependent Area under the ROC: {test_mean_auc: .4f}")



# plt.plot(event_times, test_auc, marker="o")
# plt.axhline(test_mean_auc, linestyle="--")
# plt.xlabel("Days from Enrollment")
# plt.ylabel("Time-dependent Area under the ROC")
# plt.grid(True)
# plt.show()
# plt.savefig('typical_auc_batch.png',dpi = 600)

# %% log -partial likelihood
log_lik = log_partial_lik(model_prediction.reshape(-1, 1), data_event_test.reshape(-1, 1))
print(f"Log partial likelihood: {log_lik: .4f}")


print('----------------------Fairness Measures----------------------')
# %% C_index_difference between groups for fairness measure

CI_score = CI(model_prediction, data_event_test, data_time_test, S_test[:, args.protect_index])
print(f"Concordance Imparity (CI) score (%) for test data: {CI_score: .2f}")


# c_index_group_score, c_td_index_group_score = C_index_difference(np.unique(S_test[:, args.protect_index]),
#                                                                  S_test[:, args.protect_index], data_y_train,
#                                                                  data_event_test, data_time_test, model_prediction)
# print(f"C-index group difference score (%) for test data: {c_index_group_score * 100: .2f}")
# print(f"C^td-index group difference score (%) for test data: {c_td_index_group_score * 100: .2f}")
#
# c_index_intersectional_score, c_td_index_intersectional_score = C_index_difference(intersectionalGroups, S_test,
#                                                                                    data_y_train, data_event_test,
#                                                                                    data_time_test, model_prediction)
# print(f"C-index intersectional difference score (%) for test data: {c_index_intersectional_score * 100: .2f}")
# print(f"C^td-index intersectional difference score (%) for test data: {c_td_index_intersectional_score * 100: .2f}")

# %% individual fairness measures
data_X_test_for_distance = data_X_test.numpy()
data_X_test_for_distance = data_X_test_for_distance / np.linalg.norm(data_X_test_for_distance, axis=1, keepdims=1)


if args.with_scale:
    scale_measure = 0.01
    F_I = individual_fairness_scale(model_prediction, data_X_test_for_distance, scale_measure)
    print(f"F_I individual fairness metric with scale={scale_measure: .4f}: {F_I: .4f}")
else:
    F_I = individual_fairness(model_prediction, data_X_test_for_distance)
    print(f"F_I individual fairness metric: {F_I: .4f}")

# %% group fairness measures - age or race or gender
F_G = group_fairness(model_prediction, S_test[:, args.protect_index])
print(f"F_G group fairness metric (for protect): {F_G: .4f}")


# %% intersectional fairness measures
F_S = intersect_fairness(model_prediction, S_test, intersectionalGroups)
print(f"F_S intersectional fairness metric: {F_S: .4f}")


# %% average fairness measures
F_A = (F_I + F_G + F_S) / 3
print(f"F_A average fairness metric: {F_A: .4f}")


