import torch
import torch.nn as nn
import numpy as np

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
        partial_sum_list = []
        for index in range(time.shape[0]):
            refer_time_list = refer_time.clone().tolist()
            refer_time_list.append(time[index].item())
            hazard_ratio_refer_list = hazard_ratio_refer.clone().tolist()
            hazard_ratio_refer_list.append(hazard_ratio[index].item())
            selected_time = torch.FloatTensor(refer_time_list)-time[index]
            selected_time[selected_time >= 0] = 1
            selected_time[selected_time < 0] = 0
            partial_sum_list.append(torch.sum(selected_time*torch.FloatTensor(hazard_ratio_refer_list)).item())

        log_risk = torch.log(torch.FloatTensor(partial_sum_list).view(-1, 1))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = -uncensored_likelihood.transpose(0, 1) * E.float()

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