import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torchtuples as tt
from pycox.models.utils import pad_col
from pycox.models import utils

#=======DeepHit Loss=========
def deephit_nll(out, k, t, mask1):
  I_1 = torch.sign(k)
  tmp1 = torch.sum(torch.sum(mask1 * out, 2), 1, keepdim = True)
  tmp1 = I_1 * torch.log(tmp1 + 1e-08)

  tmp2 = torch.sum(torch.sum(mask1 * out, 2), 1, keepdim = True)
  tmp2 = (1. - I_1) * torch.log(tmp2 + 1e-08)

  return torch.flatten(-(tmp1 + 1.0*tmp2))

def rank_loss(num_event, num_category, out, k, t, mask2):
  sigma1 = 0.1

  eta = []
  for e in range(num_event):
      one_vector = torch.ones_like(t, dtype = torch.float32)
      I_2 = torch.eq(k, e+1).float()
      I_2 = torch.diag_embed(torch.squeeze(I_2))
      # print(out[0:,e:e+1,0:].size())
      tmp_e = torch.reshape(out[0:,e:e+1,0:], (-1, num_category))

      R = torch.mm(tmp_e, mask2.t())

      diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
      R = torch.mm(one_vector, diag_R.t()) - R
      R = R.t()

      T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t,one_vector.t())))

      T = torch.mm(I_2, T)
      temp_eta = torch.mean(T * torch.exp(-R/sigma1), 1, keepdim = True)

      eta.append(temp_eta)

  eta = torch.stack(eta, dim = 1)
  eta = torch.mean(torch.reshape(eta, (-1, num_event)), 1, keepdim = True)

  return torch.flatten(eta)

def _reduction(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_pmf(phi, idx_durations, events, reduction= 'mean', epsilon= 1e-7):
    """Negative log-likelihood for the PMF parametrized model [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item() + 1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = utils.pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return _reduction(loss, reduction), loss

def rank_loss_deephit_dro_single_matrix(ref_predict, ref_time, phi, idx_durations, sigma, reduction= 'mean'):

    idx_durations = idx_durations.view(-1, 1)
    ref_time = ref_time.view(-1, 1)
    pmf = utils.pad_col(phi).softmax(1)
    pmf_ref = utils.pad_col(ref_predict).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)

    rank_ones_1 = torch.ones((1, pmf_ref.shape[0]), device=pmf.device)
    rank_mat_1 = idx_durations.type(torch.float).matmul(rank_ones_1)
    rank_ones_2 = torch.ones((pmf.shape[0], 1), device=pmf.device)
    rank_mat_2 = rank_ones_2.matmul(ref_time.transpose(0, 1).type(torch.float))
    rank_mat_all = ((rank_mat_1 - rank_mat_2) < 0).type(torch.float)

    n = pmf_ref.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r_ref = pmf_ref.cumsum(1).matmul(y.transpose(0, 1))
    r = ones.matmul(diag_r)-r_ref
    r = r.transpose(0, 1)
    loss = rank_mat_all * torch.exp(-r / sigma)
    loss = loss.mean(1, keepdim=True)

    return _reduction(loss, reduction), loss[:,0]



#%% Loss function is the neg log partial likelihood
class negLogLikelihood(nn.Module):
    # Source: deepSurv implementation with PyTorch (https://gitlab.com/zydou/deepsurv/-/tree/master/DeepSurv-Pytorch)
    def __init__(self):
        super(negLogLikelihood, self).__init__()

    def forward(self, prediction, targets):
        risk = prediction
        E = targets
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood.transpose(0, 1) * E.float()
        num_observed_events = torch.sum(E.float())
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events # average the loss

        return neg_likelihood

class negLogLikelihood_per_sample(nn.Module):
    # Source: deepSurv implementation with PyTorch (https://gitlab.com/zydou/deepsurv/-/tree/master/DeepSurv-Pytorch)
    def __init__(self):
        super(negLogLikelihood_per_sample, self).__init__()

    def forward(self, prediction, targets):
        risk = prediction
        E = targets
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = -uncensored_likelihood.transpose(0, 1) * E.float()

        return censored_likelihood

class negLogLikelihood_per_sample_for_splitting(nn.Module):
    # Source: deepSurv implementation with PyTorch (https://gitlab.com/zydou/deepsurv/-/tree/master/DeepSurv-Pytorch)
    def __init__(self):
        super(negLogLikelihood_per_sample_for_splitting, self).__init__()

    def forward(self, prediction, targets, time, refer_prediction, refer_time):
        risk = prediction
        E = targets
        hazard_ratio = torch.exp(risk)
        hazard_ratio_refer = torch.exp(refer_prediction[:,0])
        # partial_sum_list = []

        ones_1 = torch.ones((1,risk.shape[0]))
        mat_1 = refer_time.view(refer_time.shape[0],1).matmul(ones_1)
        ones_2 = torch.ones((refer_prediction.shape[0], 1))
        mat_2 = ones_2.matmul(time.view(time.shape[0],1).transpose(0, 1))
        mat_all = ((mat_1-mat_2)>=0).type(torch.float)

        hazard_ratio_refer_sum = hazard_ratio_refer.view(hazard_ratio_refer.shape[0], 1).transpose(0, 1).matmul(mat_all)
        partial_sum = hazard_ratio + hazard_ratio_refer_sum.transpose(0, 1)
        uncensored_likelihood = risk - torch.log(partial_sum)
        censored_likelihood = -uncensored_likelihood.transpose(0, 1) * E.float()

        # for index in range(time.shape[0]):
        #     refer_time_list = refer_time.clone().tolist()
        #     refer_time_list.append(time[index].item())
        #     hazard_ratio_refer_list = hazard_ratio_refer.clone().tolist()
        #     hazard_ratio_refer_list.append(hazard_ratio[index].item())
        #     selected_time = torch.FloatTensor(refer_time_list)-time[index]
        #     selected_time[selected_time >= 0] = 1
        #     selected_time[selected_time < 0] = 0
        #     partial_sum_list.append(torch.sum(selected_time*torch.FloatTensor(hazard_ratio_refer_list)).item())
        #
        # log_risk = torch.log(torch.FloatTensor(partial_sum_list).view(-1, 1))
        # uncensored_likelihood = risk - log_risk
        # censored_likelihood = -uncensored_likelihood.transpose(0, 1) * E.float()

        return censored_likelihood
#%% linear Cox PH model
class linearCoxPH_Regression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearCoxPH_Regression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize,bias=False)

    def forward(self, x):
        out = self.linear(x) # linear layer to output Linear to output Log Hazard Ratio
        return out

#%% Deep Cox PH model
class MLP(nn.Module):
    def __init__(self, dx, outputSize, hid=24):
        super(MLP, self).__init__()
        self.dx = dx
        self.fc1 = nn.Linear(self.dx, out_features=hid)
        self.fc2 = nn.Linear(in_features=hid, out_features=outputSize)
    def forward(self, x):
        out = nn.functional.relu(self.fc1(x.view(-1, self.dx)))
        # out = nn.functional.leaky_relu(self.fc1(x.view(-1, self.dx)))
        out2 = self.fc2(out)
        return out2

class MLP_deephit(nn.Module):
    def __init__(self, dx, outputSize, hid=32):
        super(MLP_deephit, self).__init__()
        self.dx = dx
        self.outputSize = outputSize
        self.fc1 = nn.Linear(self.dx, out_features=hid)
        self.fc2 = nn.Linear(in_features=hid, out_features=outputSize)
    def forward(self, x):
        out = nn.functional.relu(self.fc1(x.view(-1, self.dx)))
        # out = nn.functional.leaky_relu(self.fc1(x.view(-1, self.dx)))
        out2 = self.fc2(out)
        out2 = F.softmax(out2, -1)
        out2 = out2.reshape((-1, 1, self.outputSize))
        return out2

# class MLP(nn.Module):
#     def __init__(self, dx, outputSize,hid=256):
#         super(MLP, self).__init__()
#         self.dx = dx
#         self.fc1 = nn.Linear(self.dx, out_features=32)
#         self.fc2 = nn.Linear(in_features=32, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=64)
#         self.fc5 = nn.Linear(in_features=64, out_features=outputSize)
#     def forward(self, x):
#         out = nn.functional.relu(self.fc1(x.view(-1, self.dx)))
#         out1 = nn.functional.relu(self.fc2(out))
#         out2 = nn.functional.relu(self.fc3(out1))
#         out3 = nn.functional.relu(self.fc4(out2))
#         # out = nn.functional.leaky_relu(self.fc1(x.view(-1, self.dx)))
#         out4 = self.fc5(out3)
#         return out4


def predict_surv(input, predict_np):
    pmf = predict_pmf(input, predict_np)
    surv = 1 - pmf.cumsum(1)
    return tt.utils.array_or_tensor(surv, True, input)

def predict_pmf(input, predict_np):
    preds = predict_np
    pmf = pad_col(torch.from_numpy(preds)).softmax(1)[:, :-1]
    return tt.utils.array_or_tensor(pmf, True, input)

def predict_surv_df(input, predict_np, duration_index):
    surv = predict_surv(input, predict_np)
    return pd.DataFrame(surv.transpose(), duration_index)