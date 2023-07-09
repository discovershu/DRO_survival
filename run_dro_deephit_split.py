import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %% dataset pre-processing

from performance_measures import c_index, brier_score, weighted_c_index, weighted_brier_score, log_partial_lik
from neural_models import negLogLikelihood, linearCoxPH_Regression, MLP, negLogLikelihood_per_sample, nll_pmf, \
    rank_loss_deephit_dro_single_matrix, predict_surv_df
from fairness_measures import individual_fairness, group_fairness, intersect_fairness, individual_fairness_scale, CI, \
    C_index_difference, individual_fairness_scale_censoring, group_fairness_censoring, CI_td, \
    individual_fairness_scale_censoring_td, group_fairness_censoring_td

from sksurv.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle
from deephit_model import DeepHitSingle_dro, DeepHitSingle_dro_split
import torchtuples as tt  # Some useful functions
import os
# %% linear Cox PH model in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
from scipy import optimize
import datetime

parser = argparse.ArgumentParser(description='PyTorch Joint DRO for Survival Analysis')
parser.add_argument('--dataset', type=str, default="SEER2",
                    help="choose from 'FLC', 'SUPPORT', 'COMPAS', 'SEER', 'SEER2'")
parser.add_argument('--model', type=str, default="MLP", help="choose from 'Linear' and 'MLP'")
parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--with_scale', type=bool, default=True, help='use scale for testing')
parser.add_argument('--eps', type=float, default=0.5,
                    help='a lower bound on the group proportions e.g. {0.05, 0.1, 0.15, 0.2, 0.3}')
parser.add_argument('--split', type=float, default=0.5,
                    help='Splitting training set to two parts. This value is the prop. of second training set for calculating logsumexp')
parser.add_argument('--seed', type=int, default=7, help='Seed')
parser.add_argument('--protect_index', type=int, default=0, help='protect attribute index')
parser.add_argument('--train_or_evaluation', type=int, default=0, help='0 for training, 1 for evaluation')
parser.add_argument('--deephit_alpha', type=float, default=0.2, metavar='da', help='deephit alpha')

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

num_durations = int(int(np.max(data_y['futime'])) * 1.0)
labtrans = DeepHitSingle.label_transform(num_durations)

# train-test split
data_X_train, data_X_test, data_y_train, data_y_test, S_train, S_test = train_test_split(data_x, data_y, protect_attr,
                                                                                         test_size=0.2,
                                                                                         stratify=data_y["death"],
                                                                                         random_state=7)
data_X_train, data_X_dev, data_y_train, data_y_dev, S_train, S_dev = train_test_split(data_X_train, data_y_train,
                                                                                      S_train, test_size=0.2,
                                                                                      stratify=data_y_train["death"],
                                                                                      random_state=args.seed)
data_X_train_1, data_X_train_2, data_y_train_1, data_y_train_2, S_train_1, S_train_2 = train_test_split(data_X_train,
                                                                                                        data_y_train,
                                                                                                        S_train,
                                                                                                        test_size=args.split,
                                                                                                        stratify=
                                                                                                        data_y_train[
                                                                                                            "death"],
                                                                                                        random_state=args.seed)
#
data_X_train, data_event_train, data_time_train = check_arrays_survival(data_X_train, data_y_train)
data_X_train, data_event_train, data_time_train, S_train = prepare_data(data_X_train, data_event_train, data_time_train,
                                                                        S_train)
data_X_train_1, data_event_train_1, data_time_train_1 = check_arrays_survival(data_X_train_1, data_y_train_1)
data_X_train_2, data_event_train_2, data_time_train_2 = check_arrays_survival(data_X_train_2, data_y_train_2)
# data_X_train_1, data_event_train_1, data_time_train_1, S_train_1 = prepare_data(data_X_train_1, data_event_train_1, data_time_train_1, S_train_1)
# data_X_train_2, data_event_train_2, data_time_train_2, S_train_2 = prepare_data(data_X_train_2, data_event_train_2, data_time_train_2, S_train_2)


y_train = labtrans.fit_transform(*(data_time_train, data_event_train))
data_time_train = y_train[0]
data_event_train = y_train[1]
train = (np.float32(data_X_train), y_train)
y_train_1 = labtrans.transform(*(data_time_train_1, data_event_train_1))
data_time_train_1 = y_train_1[0]
data_event_train_1 = y_train_1[1]
train_1 = (np.float32(data_X_train_1), y_train_1)
y_train_2 = labtrans.transform(*(data_time_train_2, data_event_train_2))
data_time_train_2 = y_train_2[0]
data_event_train_2 = y_train_2[1]
train_2 = (np.float32(data_X_train_2), y_train_2)
val = (data_X_dev, data_y_dev)  ###initial but not use

if args.train_or_evaluation == 0:
    data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_dev, data_y_dev)
    data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test,
                                                                        S_dev)
    data_y_test = data_y_dev
    y_val = labtrans.transform(*(data_time_test, data_event_test))
    val = (np.float32(data_X_test), y_val)
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

# %%
# hyperparameters of the model
input_size = data_X_train.shape[1]
out_features = labtrans.out_features

# %% intialize model and optimizar
# initialize cox PH model
if args.model == 'Linear':
    deephit_model = linearCoxPH_Regression(input_size, out_features)
if args.model == 'MLP':
    num_nodes = [32, 32]
    # num_nodes = [256]
    batch_norm = True
    dropout = 0.1
    deephit_model = tt.practical.MLPVanilla(input_size, num_nodes, out_features, batch_norm, dropout)

# %% training cox ph model
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

# Loss and optimizer

optimizer = optim.Adam(deephit_model.parameters(), lr=args.lr)

saved_model_name = "saved_models/DRO_SPLIT_deephit_{}_{}_lr_{}_seed_{}_sensitive_{}_alpha_{}" \
    .format(args.dataset, args.model, args.lr, args.seed, 0, args.eps)
para_init = deephit_model.state_dict()
model_prediction = predict_surv_df(np.float32(data_X_test), deephit_model(data_X_test).detach().numpy(),
                                   labtrans.cuts)
if args.train_or_evaluation == 0:
    print('Model Training')
    # %% save the model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    best_c_index = 0.
    starttime = datetime.datetime.now()
    flag = 0
    for epoch in range(args.epochs):
        outputs_1 = deephit_model(data_X_train_1)
        outputs_2 = deephit_model(data_X_train_2)

        nll_1, per_sample_nll_1 = nll_pmf(outputs_1, data_time_train_1.to(torch.long), data_event_train_1,
                                          reduction='mean')
        rank_loss_1, per_sample_rank_loss_1 = rank_loss_deephit_dro_single_matrix(outputs_2,
                                                                                  data_time_train_2.to(torch.long),
                                                                                  outputs_1,
                                                                                  data_time_train_1.to(torch.long),
                                                                                  0.1, reduction='mean')
        per_sample_losses_1 = args.deephit_alpha * per_sample_nll_1 + (1. - args.deephit_alpha) * per_sample_rank_loss_1
        chi_loss_np_1 = loss_map_chi_factory(per_sample_losses_1.detach().numpy(), args.eps)
        cutpt_1 = optimize.fminbound(chi_loss_np_1, np.min(per_sample_losses_1.detach().numpy()) - 1000.0,
                                     np.max(per_sample_losses_1.detach().numpy()))
        loss_1 = loss_map_chi_factory_tensor(per_sample_losses_1, args.eps, cutpt_1)

        nll_2, per_sample_nll_2 = nll_pmf(outputs_2, data_time_train_2.to(torch.long), data_event_train_2,
                                          reduction='mean')
        rank_loss_2, per_sample_rank_loss_2 = rank_loss_deephit_dro_single_matrix(outputs_1,
                                                                                  data_time_train_1.to(torch.long),
                                                                                  outputs_2,
                                                                                  data_time_train_2.to(torch.long),
                                                                                  0.1, reduction='mean')
        per_sample_losses_2 = args.deephit_alpha * per_sample_nll_2 + (1. - args.deephit_alpha) * per_sample_rank_loss_2
        chi_loss_np_2 = loss_map_chi_factory(per_sample_losses_2.detach().numpy(), args.eps)
        cutpt_2 = optimize.fminbound(chi_loss_np_2, np.min(per_sample_losses_2.detach().numpy()) - 1000.0,
                                     np.max(per_sample_losses_2.detach().numpy()))
        loss_2 = loss_map_chi_factory_tensor(per_sample_losses_2, args.eps, cutpt_2)

        loss = loss_1 + loss_2

        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()
        optimizer.step()

        model_prediction_mid = predict_surv_df(np.float32(data_X_test), deephit_model(data_X_test).detach().numpy(),
                                               labtrans.cuts)
        ev_mid = EvalSurv(model_prediction_mid, data_time_test, data_event_test.astype(int), censor_surv='km')
        ctd = ev_mid.concordance_td('antolini')
        # print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()), f"C^td-index: {ctd: .4f}")

        if ctd > best_c_index:
            flag = 0
            # print('epoch:', epoch)
            best_c_index = ctd
            model_prediction = model_prediction_mid
            torch.save(deephit_model.state_dict(), saved_model_name)
        else:
            flag = flag + 1
            if flag >= 50:
                break

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Training Time:', time)

else:
    print('Load the Saved Model')
    deephit_model.load_state_dict(torch.load(saved_model_name))
    model_prediction = predict_surv_df(np.float32(data_X_test), deephit_model(data_X_test).detach().numpy(),
                                       labtrans.cuts)

print('----------------------Accuracy Measures----------------------')

ev = EvalSurv(model_prediction, data_time_test, data_event_test.astype(int), censor_surv='km')

print(f"Pycox C^td-index for test data: {ev.concordance_td('antolini'): .4f}")

time_grid = np.linspace(data_time_test.min(), data_time_test.max(), 100)
print("integrated brier score from Pycox: {:0.4f}".format(ev.integrated_brier_score(time_grid)))

print(f"binomial_log_likelihood for test data: {-ev.integrated_nbll(time_grid): .4f}")

print('----------------------Fairness Measures----------------------')
# %% C_index_difference between groups for fairness measure

CI_score = CI_td(model_prediction.values, data_event_test, data_time_test, S_test[:, args.protect_index])
print(f"Concordance Imparity (CI) score (%) for test data: {CI_score: .2f}")

F_CI = individual_fairness_scale_censoring_td(model_prediction.values, data_X_test, 0.0001, data_event_test,
                                              data_time_test)
print(f"F_CI individual censoring fairness metric with scale={0.01: .4f}: {F_CI: .4f}")

# %% group fairness measures - age or race or gender
F_CG = group_fairness_censoring_td(model_prediction.values, S_test[:, args.protect_index], data_X_test, 0.0001,
                                   data_event_test, data_time_test)
print(f"F_CG group censoring fairness metric (for protect): {F_CG: .4f}")



