import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %% dataset pre-processing

from performance_measures import c_index, brier_score, weighted_c_index, weighted_brier_score, log_partial_lik
from neural_models import negLogLikelihood, linearCoxPH_Regression, MLP, negLogLikelihood_per_sample
from fairness_measures import individual_fairness, group_fairness, individual_fairness_td, group_fairness_td, \
    individual_fairness_scale_td, intersect_fairness, individual_fairness_scale, CI, \
    C_index_difference, individual_fairness_scale_censoring, group_fairness_censoring, \
    individual_fairness_scale_censoring_td, group_fairness_censoring_td

from sksurv.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
import os
# %% linear Cox PH model in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
from scipy import optimize
import datetime
from pycox.evaluation import EvalSurv

parser = argparse.ArgumentParser(description='PyTorch Joint DRO for Survival Analysis')
parser.add_argument('--dataset', type=str, default="FLC", help="choose from 'FLC', 'SUPPORT', 'COMPAS', 'SEER'")
parser.add_argument('--model', type=str, default="Linear", help="choose from 'Linear' and 'MLP'")
parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
parser.add_argument('--epochs', type=int, default=5000, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--with_scale', type=bool, default=True, help='use scale for testing')
parser.add_argument('--eps', type=float, default=0.9,
                    help='a lower bound on the group proportions e.g. {0.05, 0.1, 0.15, 0.2, 0.3}')
parser.add_argument('--seed', type=int, default=7, help='Seed')
parser.add_argument('--protect_index', type=int, default=0, help='protect attribute index')
parser.add_argument('--train_or_evaluation', type=int, default=1,
                    help='0 for training, 1 for evaluation on test, 2 for evaluation on dev')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
out_str = str(args)
print(out_str)

# %%
from compute_survival_function import predict_survival_function


# The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)


def threshplus(x):
    y = x.copy()
    y[y < 0] = 0
    return y


def threshplus_tensor(x):
    y = x.clone()
    y[y < 0] = 0
    return y


def loss_map_chi_factory(loss_values, eps):
    # return lambda x: np.sqrt(2)*(1.0/eps-1.0)*np.sqrt(np.mean(threshplus(loss_values-x)**2.0)) + x
    return lambda x: np.sqrt(2 * ((1.0 / eps - 1.0) ** 2.0) + 1) * np.sqrt(
        np.mean(threshplus(loss_values - x) ** 2.0)) + x


def loss_map_chi_factory_tensor(loss_values, eps, opt_eta):
    # return np.sqrt(2)*(1.0/eps-1.0)*torch.sqrt(torch.mean(threshplus_tensor(loss_values-opt_eta)**2.0)) + opt_eta
    return np.sqrt(2 * ((1.0 / eps - 1.0) ** 2.0) + 1) * torch.sqrt(
        torch.mean(threshplus_tensor(loss_values - opt_eta) ** 2.0)) + opt_eta


# %% FLC data:
from utilities import prepare_data
from utilities import check_arrays_survival
from flc_data_preprocess import flc_preprocess
from support_data_preprocess import support_preprocess
from compas_data_preprocess import compas_preprocess
from seer_data_preprocess import seer_preprocess
from seer2_data_preprocess import seer2_preprocess

# Survival Data
if args.dataset == 'FLC':
    data_x, data_y, protect_attr = flc_preprocess()
elif args.dataset == 'SUPPORT':
    data_x, data_y, protect_attr = support_preprocess()
elif args.dataset == 'COMPAS':
    data_x, data_y, protect_attr = compas_preprocess()
elif args.dataset == 'SEER':
    data_x, data_y, protect_attr = seer_preprocess()
elif args.dataset == 'SEER2':
    data_x, data_y, protect_attr = seer2_preprocess()
else:
    print('unknown')

# train-test split
data_X_train, data_X_test, data_y_train, data_y_test, S_train, S_test = train_test_split(data_x, data_y, protect_attr,
                                                                                         test_size=0.2,
                                                                                         stratify=data_y["death"],
                                                                                         random_state=7)
data_X_train, data_X_dev, data_y_train, data_y_dev, S_train, S_dev = train_test_split(data_X_train, data_y_train,
                                                                                      S_train, test_size=0.2,
                                                                                      stratify=data_y_train["death"],
                                                                                      random_state=args.seed)
#
data_X_train, data_event_train, data_time_train = check_arrays_survival(data_X_train, data_y_train)
data_X_train, data_event_train, data_time_train, S_train = prepare_data(data_X_train, data_event_train, data_time_train,
                                                                        S_train)

if args.train_or_evaluation == 0 or args.train_or_evaluation == 2:
    data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_dev, data_y_dev)
    data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test,
                                                                        S_dev)
    data_y_test = data_y_dev
else:
    data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_test, data_y_test)
    data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test,
                                                                        S_test)
#
intersectionalGroups = np.unique(S_train, axis=0)  # all intersecting groups, i.e. black-women, white-man etc
# data normalization: mean subtraction method to compute euclidean distance
scaler = StandardScaler()
scaler.fit(data_X_train)
data_X_train = scaler.transform(data_X_train)
data_X_test = scaler.transform(data_X_test)

# unique times
selected_values = data_time_train[data_event_train]
unique_time = np.unique(selected_values)
if 0 not in unique_time:
    unique_time = np.insert(unique_time, 0, 0.0)
unique_time_pre = np.insert(unique_time[:-1], 0, 0.0)
diff = unique_time - unique_time_pre

# kappa
kappa = np.zeros(len(data_event_train), dtype=int)
for i in range(len(data_event_train)):
    if data_event_train[i] == True:
        kappa[i] = np.where(unique_time == data_time_train[i])[0]
    else:
        if data_time_train[i] == 0:
            kappa[i] = 0
        else:
            selected_indexes = np.where(unique_time < data_time_train[i])[0]
            kappa[i] = selected_indexes[-1]

# kappa for test
kappa_test = np.zeros(len(data_time_test), dtype=int)
for i in range(len(data_time_test)):
    if data_time_test[i] in unique_time:
        kappa_test[i] = np.where(unique_time == data_time_test[i])[0]
    else:
        selected_indexes_test = np.where(unique_time < data_time_test[i])[0]
        kappa_test[i] = selected_indexes_test[-1]

# psi
psi_params = nn.Parameter(torch.randn(len(unique_time)), requires_grad=True)

# %%
# hyperparameters of the model
input_size = data_X_train.shape[1]
output_size = 1

# %% intialize model and optimizar
# initialize cox PH model
if args.model == 'Linear':
    coxPH_model = linearCoxPH_Regression(input_size, output_size)
if args.model == 'MLP':
    coxPH_model = MLP(input_size, output_size)
# Loss and optimizer
criterion = negLogLikelihood()
criterion_per_sample = negLogLikelihood_per_sample()

optimizer = optim.Adam([{'params': coxPH_model.parameters()}, {'params': psi_params}], lr=args.lr)

# %% training cox ph model
data_X_train = Variable((torch.from_numpy(data_X_train)).float())
data_event_train = Variable((torch.from_numpy(data_event_train)).float())
data_time_train = Variable((torch.from_numpy(data_time_train)).float())

data_X_test = Variable((torch.from_numpy(data_X_test)).float())

saved_model_name = "/mnt/hdd/project/shuhu/shu/DROSurv/saved_models/DRO_COX_FULL_{}_{}_lr_{}_seed_{}_sensitive_{}_alpha_{}" \
    .format(args.dataset, args.model, args.lr, args.seed, args.protect_index, args.eps)
saved_model_psi_name = "/mnt/hdd/project/shuhu/shu/DROSurv/saved_models/DRO_COX_FULL_psi_{}_{}_lr_{}_seed_{}_sensitive_{}_alpha_{}" \
    .format(args.dataset, args.model, args.lr, args.seed, args.protect_index, args.eps)
if args.train_or_evaluation == 0:
    print('Model Training')
    starttime = datetime.datetime.now()
    # %% stochastic method
    for epoch in range(args.epochs):
        # for batch in range(0,np.int64(np.floor(len(data_X_train)/mini_batch))*mini_batch,mini_batch):

        # backward propagation
        outputs = coxPH_model(data_X_train)
        psi_values = torch.tensor([psi_params[i] for i in kappa]).unsqueeze(-1)
        first_term = (-1) * data_event_train.unsqueeze(-1) * (outputs + psi_values)

        range_tensor = torch.arange(len(unique_time)).unsqueeze(1).expand(-1, len(kappa))
        mask_matrix = (range_tensor <= torch.from_numpy(kappa)).float()
        inner_sum_values = torch.from_numpy(diff) * torch.exp(psi_params)
        inner_sum = torch.matmul(inner_sum_values.unsqueeze(0).to(dtype=torch.float32), mask_matrix)
        second_term = inner_sum.t() * torch.exp(outputs)

        per_sample_losses = (first_term + second_term).t()

        chi_loss_np = loss_map_chi_factory(per_sample_losses.detach().numpy(), args.eps)
        cutpt = optimize.fminbound(chi_loss_np, np.min(per_sample_losses.detach().numpy()) - 1000.0,
                                   np.max(per_sample_losses.detach().numpy()))
        loss = loss_map_chi_factory_tensor(per_sample_losses, args.eps, cutpt)

        # loss = torch.mean(per_sample_losses)

        optimizer.zero_grad()  # zero the parameter gradients
        # optimizer2.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer2.step()
        # print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()))

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Training Time:', time)

    # %% save the model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(coxPH_model.state_dict(), saved_model_name)
    torch.save(psi_params, saved_model_psi_name)
else:
    print('Load the Saved Model')
    coxPH_model.load_state_dict(torch.load(saved_model_name))
    psi_params = torch.load(saved_model_psi_name)

# %% Evaluate the model

with torch.no_grad():
    base_prediction = coxPH_model(data_X_train)
    base_prediction = (base_prediction.numpy()).reshape((-1,))  # beta \dot x
# linear predictor for test/dev data
with torch.no_grad():
    model_prediction = coxPH_model(data_X_test)
    model_prediction = (model_prediction.numpy()).reshape((-1,))  # beta \dot x

print('----------------------Accuracy Measures----------------------')

skSurv_result_test = concordance_index_censored(data_event_test, data_time_test, model_prediction)
print(f"skSurv implemented C-index for test data: {skSurv_result_test[0]: .4f}")

data_event_train = data_event_train.numpy().astype(bool)
data_time_train = data_time_train.numpy()

mask_test = torch.triu(torch.ones_like(torch.zeros(diff.size, diff.size)))
inner_sum_values_test = torch.from_numpy(diff) * torch.exp(psi_params)
inner_sum_test = -torch.matmul(inner_sum_values_test.unsqueeze(0).to(dtype=torch.float32), mask_test)
predict_surv = torch.exp(torch.exp(torch.tensor(model_prediction).unsqueeze(1)) * inner_sum_test)
predict_surv_df = pd.DataFrame(predict_surv.detach().numpy().transpose())

ev = EvalSurv(predict_surv_df, data_time_test, data_event_test.astype(int), censor_surv='km')
print(f"Pycox C^td-index for test data: {ev.concordance_td('antolini'): .4f}")

time_grid = np.linspace(data_time_test.min(), data_time_test.max(), 100)
print("integrated brier score from Pycox: {:0.4f}".format(ev.integrated_brier_score(time_grid)))

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

# %% log -partial likelihood
log_lik = log_partial_lik(model_prediction.reshape(-1, 1), data_event_test.reshape(-1, 1))
print(f"Log partial likelihood: {log_lik: .4f}")

print('----------------------Fairness Measures----------------------')
# %% C_index_difference between groups for fairness measure
CI_score = CI(model_prediction, data_event_test, data_time_test, S_test[:, args.protect_index])
print(f"Concordance Imparity (CI) score (%) for test data: {CI_score: .2f}")

# %% individual fairness measures
data_X_test_for_distance = data_X_test.numpy()
data_X_test_for_distance = data_X_test_for_distance / np.linalg.norm(data_X_test_for_distance, axis=1, keepdims=1)

if args.with_scale:
    scale_measure = 0.01
    F_I = individual_fairness_scale_td(predict_surv_df.values, data_X_test.numpy(), scale_measure)
    print(f"F_I individual fairness metric with scale={scale_measure: .4f}: {F_I: .20f}")
else:
    F_I = individual_fairness_td(predict_surv_df.values, data_X_test_for_distance)
    print(f"F_I individual fairness metric: {F_I: .20f}")

F_CI = individual_fairness_scale_censoring_td(predict_surv_df.values, data_X_test.numpy(), 0.01, data_event_test,
                                              data_time_test)
print(f"F_CI individual censoring fairness metric with scale={0.01: .4f}: {F_CI: .20f}")

# %% group fairness measures - age or race or gender
F_G = group_fairness_td(predict_surv_df.values, S_test[:, args.protect_index])
print(f"F_G group fairness metric (for protect): {F_G: .20f}")

F_CG = group_fairness_censoring_td(predict_surv_df.values, S_test[:, args.protect_index], data_X_test.numpy(), 0.01,
                                   data_event_test, data_time_test)
print(f"F_CG group censoring fairness metric (for protect): {F_CG: .20f}")



