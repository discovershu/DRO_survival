import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
import torch


# %% fairness metric: individual fairness
def individual_fairness(prediction, X):
    HazardFunction = np.exp(prediction)
    # =============================================================================
    #     normalizer = np.linalg.norm(model_prediction, 2) # L2 normalization
    #     HazardFunction = HazardFunction/normalizer # L2 normalization
    # =============================================================================
    N = len(prediction)
    R_beta = 0.0  # initialization of individual fairnessd
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j <= i:
                continue
            else:
                distance = np.sqrt(sum((X[i] - X[j]) ** 2))  # euclidean distance
                R_beta = R_beta + max(0, (np.abs(HazardFunction[i] - HazardFunction[j]) - distance))
    R_beta_avg = R_beta / (N * (N - 1))
    return R_beta_avg


def individual_fairness_td(prediction, X):
    F_I_score_list = []
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]
    for index in eval_index:
        HazardFunction = prediction[index]
        # =============================================================================
        #     normalizer = np.linalg.norm(model_prediction, 2) # L2 normalization
        #     HazardFunction = HazardFunction/normalizer # L2 normalization
        # =============================================================================
        N = len(prediction[index])
        R_beta = 0.0  # initialization of individual fairnessd
        for i in range(len(prediction[index])):
            for j in range(len(prediction[index])):
                if j <= i:
                    continue
                else:
                    distance = np.sqrt(sum((X[i] - X[j]) ** 2))  # euclidean distance
                    R_beta = R_beta + max(0, (np.abs(HazardFunction[i] - HazardFunction[j]) - distance))
        R_beta_avg = R_beta / (N * (N - 1))
        F_I_score_list.append(R_beta_avg)
    F_I_score = np.mean(np.asarray(F_I_score_list))
    return F_I_score


def individual_fairness_scale(prediction, X, scale):
    HazardFunction = np.exp(prediction)
    N = len(prediction)
    R_beta = 0.0  # initialization of individual fairnessd
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j <= i:
                continue
            else:
                distance = np.sqrt(sum((X[i] - X[j]) ** 2))  # euclidean distance
                R_beta = R_beta + max(0, (np.abs(HazardFunction[i] - HazardFunction[j]) - scale * distance))
    R_beta_avg = R_beta / (N * (N - 1))
    return R_beta_avg


def individual_fairness_scale_td(prediction, X, scale):
    F_I_score_list = []
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]
    for index in eval_index:
        N = len(prediction[index])
        mat_select = np.zeros((N, N))
        rows, cols = np.triu_indices(N, k=1)
        mat_select[rows, cols] = 1
        mat_select = torch.from_numpy(mat_select)

        diff_pred = torch.from_numpy(prediction[index]).unsqueeze(1) - torch.from_numpy(prediction[index]).unsqueeze(0)
        distances_pred = torch.abs(diff_pred)

        diff = torch.from_numpy(X).unsqueeze(1) - torch.from_numpy(X).unsqueeze(0)
        distances_squared = torch.sum(torch.square(diff), dim=-1)
        distances = torch.sqrt(distances_squared)

        F_CI_score = torch.sum(torch.relu(distances_pred - scale * distances) * mat_select).detach().numpy()
        F_CI_score = F_CI_score / (N * (N - 1))
        F_I_score_list.append(F_CI_score)
    F_I_score = np.mean(np.asarray(F_I_score_list))
    return F_I_score


def individual_fairness_scale_censoring(prediction, X, scale, event, time):
    HazardFunction = np.exp(prediction)
    R_beta = 0.0  # initialization of individual fairnessd
    zeroTerm = 0.0
    censoring_count = np.sum(event == 0)
    uncensored_count = np.sum(event == 1)
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if event[i] == 0:
                if event[j] == 1:
                    if time[i] <= time[j]:
                        distance = np.sqrt(sum((X[i] - X[j]) ** 2))
                        R_beta = R_beta + max(zeroTerm, (
                                np.abs(HazardFunction[i] - HazardFunction[j]) - scale * distance))
    if censoring_count != 0 and uncensored_count != 0:
        R_beta_avg = R_beta / (censoring_count * uncensored_count)
    else:
        R_beta_avg = R_beta
    return R_beta_avg


def individual_fairness_scale_censoring_td(prediction, X, scale, event, time):
    c_ref_time = torch.from_numpy(time[event == False])
    uc_ref_time = torch.from_numpy(time[event == True])
    c_surv = torch.from_numpy(prediction[:, event == False])
    uc_surv = torch.from_numpy(prediction[:, event == True])
    c_X = torch.from_numpy(X[event == False])
    uc_X = torch.from_numpy(X[event == True])
    censoring_count = np.sum(event == False)
    uncensored_count = np.sum(event == True)
    F_CI_score_list = []
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]

    for index in eval_index:
        diff_time = c_ref_time.unsqueeze(1) - uc_ref_time.unsqueeze(0)
        mat_all = (diff_time <= 0).type(torch.float)

        diff_pred = c_surv[index, :].unsqueeze(1) - uc_surv[index, :].unsqueeze(0)
        distances_pred = torch.abs(diff_pred)

        diff = c_X.unsqueeze(1) - uc_X.unsqueeze(0)
        distances_squared = torch.sum(torch.square(diff), dim=-1)
        distances = torch.sqrt(distances_squared)

        F_CI_score = torch.sum(torch.relu(distances_pred - scale * distances) * mat_all).detach().numpy()
        F_CI_score = F_CI_score / (censoring_count * uncensored_count)
        F_CI_score_list.append(F_CI_score)
    F_CI_score = np.mean(np.asarray(F_CI_score_list))
    return F_CI_score


# %% fairness metric: group fairness
def group_fairness(prediction, S):
    h_ratio = np.exp(prediction)
    unique_group = np.unique(S)
    avg_h_ratio = sum(h_ratio) / len(h_ratio)

    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses

    h_ratio_group = np.zeros((len(unique_group)))
    group_total = np.zeros((len(unique_group)))

    for i in range(len(h_ratio)):
        h_ratio_group[S[i]] = h_ratio_group[S[i]] + h_ratio[i]
        group_total[S[i]] = group_total[S[i]] + 1

    avg_h_ratio_group = (h_ratio_group + dirichletAlpha) / (group_total + concentrationParameter)

    group_fairness = np.max(np.abs(avg_h_ratio_group - avg_h_ratio))
    return group_fairness


def group_fairness_td(prediction, S):
    F_G_score_list = []
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]
    for index in eval_index:
        h_ratio = prediction[index]
        unique_group = np.unique(S)
        avg_h_ratio = sum(h_ratio) / len(h_ratio)

        numClasses = 2
        concentrationParameter = 1.0
        dirichletAlpha = concentrationParameter / numClasses

        h_ratio_group = np.zeros((len(unique_group)))
        group_total = np.zeros((len(unique_group)))

        for i in range(len(h_ratio)):
            h_ratio_group[S[i]] = h_ratio_group[S[i]] + h_ratio[i]
            group_total[S[i]] = group_total[S[i]] + 1

        avg_h_ratio_group = (h_ratio_group + dirichletAlpha) / (group_total + concentrationParameter)

        group_fairness = np.max(np.abs(avg_h_ratio_group - avg_h_ratio))
        F_G_score_list.append(group_fairness)
    F_G_score = np.mean(np.asarray(F_G_score_list))
    return F_G_score


def group_fairness_censoring(prediction, S, X, scale, event, time):
    h_ratio = np.exp(prediction)
    unique_group = np.unique(S)
    zeroTerm = 0.0

    h_ratio_group = np.zeros((len(unique_group)))

    censoring_count = np.sum(event == 0)
    uncensored_count = np.sum(event == 1)

    for i in range(len(h_ratio)):
        for j in range(len(h_ratio)):
            if event[i] == 0 and event[j] == 1 and S[i] == S[j] and time[i] <= time[j]:
                distance = np.sqrt(sum((X[i] - X[j]) ** 2))
                h_ratio_group[S[i]] = h_ratio_group[S[i]] + max(zeroTerm,
                                                                (np.abs(h_ratio[i] - h_ratio[
                                                                    j]) - scale * distance))

    if censoring_count != 0 and uncensored_count != 0:
        group_fairness = np.sum(h_ratio_group) / (censoring_count * uncensored_count)
    else:
        group_fairness = zeroTerm
    return group_fairness


def group_fairness_censoring_td(prediction, S, X, scale, event, time):
    unique_group = np.unique(S)
    censoring_count = np.sum(event == False)
    uncensored_count = np.sum(event == True)
    F_CG_score = 0
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]

    for group in unique_group:
        group_surv = prediction[:, S == group]
        group_ref_event = event[S == group]
        group_ref_time = time[S == group]
        c_surv = torch.from_numpy(group_surv[:, group_ref_event == 0])
        c_ref_time = torch.from_numpy(group_ref_time[group_ref_event == 0])
        uc_surv = torch.from_numpy(group_surv[:, group_ref_event == 1])
        uc_ref_time = torch.from_numpy(group_ref_time[group_ref_event == 1])

        grou_X = X[S == group]
        c_X = torch.from_numpy(grou_X[group_ref_event == 0])
        uc_X = torch.from_numpy(grou_X[group_ref_event == 1])

        F_CG_score_group = 0

        for index in eval_index:
            diff_time = c_ref_time.unsqueeze(1) - uc_ref_time.unsqueeze(0)
            mat_all = (diff_time <= 0).type(torch.float)

            diff_pred = c_surv[index, :].unsqueeze(1) - uc_surv[index, :].unsqueeze(0)
            distances_pred = torch.abs(diff_pred)

            diff = c_X.unsqueeze(1) - uc_X.unsqueeze(0)
            distances_squared = torch.sum(torch.square(diff), dim=-1)
            distances = torch.sqrt(distances_squared)

            F_CG_score_group = F_CG_score_group + torch.sum(
                torch.relu(distances_pred - scale * distances) * mat_all).detach().numpy()

        F_CG_score = F_CG_score + F_CG_score_group

    F_CG_score_avg = F_CG_score / (censoring_count * uncensored_count * len(eval_index))
    return F_CG_score_avg

    # unique_group = np.unique(S)
    # F_CG_score_sum_group = np.zeros((len(unique_group)))
    # censoring_count = np.sum(event == False)
    # uncensored_count = np.sum(event == True)
    # F_CG_score_list = []
    # eval_index = [int(np.percentile(range(prediction.shape[0]), 25)), int(np.percentile(range(prediction.shape[0]), 50)),
    #               int(np.percentile(range(prediction.shape[0]), 75))]
    # for index in eval_index:
    #     for i in range(len(prediction[0])):
    #         for j in range(len(prediction[0])):
    #             if event[i] == False and event[j] == True and S[i] == S[j] and time[i] <= time[j]:
    #                 distance = np.sqrt(sum((X[i] - X[j]) ** 2))
    #                 F_CG_score_sum_group[int(S[i])] = F_CG_score_sum_group[int(S[i])] + max(0.0, (
    #                         np.abs(prediction[index][i] - prediction[index][j]) - scale * distance))
    #     if censoring_count != 0 and uncensored_count != 0:
    #         F_CG_score_avg = np.sum(F_CG_score_sum_group) / (censoring_count * uncensored_count)
    #     else:
    #         F_CG_score_avg = 0.0
    #     F_CG_score_list.append(F_CG_score_avg)
    # F_CG_score = np.mean(np.asarray(F_CG_score_list))
    # return F_CG_score


# %% fairness metric: intersectional fairness
def intersect_fairness(prediction, S, intersect_groups):
    h_ratio = np.exp(prediction)

    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses

    h_ratio_group = np.zeros((len(intersect_groups)))
    group_total = np.zeros((len(intersect_groups)))

    for i in range(len(h_ratio)):
        index = np.where((intersect_groups == S[i]).all(axis=1))[0][0]
        h_ratio_group[index] = h_ratio_group[index] + h_ratio[i]
        group_total[index] = group_total[index] + 1

    avg_h_ratio_group = (h_ratio_group + dirichletAlpha) / (group_total + concentrationParameter)

    epsilon = 0.0  # intersectional fairness
    for i in range(len(avg_h_ratio_group)):
        for j in range(len(avg_h_ratio_group)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon, abs(np.log(avg_h_ratio_group[i]) - np.log(avg_h_ratio_group[j])))
    return epsilon


def CI(prediction, data_event, data_time, S):
    C_group = np.zeros(len(np.unique(S)), dtype=float)
    P_group = np.zeros(len(np.unique(S)), dtype=float)

    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j == i:
                continue
            else:
                if ((data_time[i] < data_time[j]) and data_event[i] == False) \
                        or ((data_time[i] > data_time[j]) and data_event[j] == False) \
                        or ((data_time[i] == data_time[j]) and (data_event[i] == False and data_event[j] == False)):
                    continue
                else:
                    P_group[S[i]] = P_group[S[i]] + 1.0
                if data_time[i] < data_time[j]:
                    if prediction[i] > prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif prediction[i] == prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 0.5
                elif data_time[i] > data_time[j]:
                    if prediction[i] < prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif prediction[i] == prediction[j]:
                        C_group[S[i]] = C_group[S[i]] + 0.5
                elif data_time[i] == data_time[j]:
                    if data_event[i] == True and data_event[j] == True:
                        if prediction[i] == prediction[j]:
                            C_group[S[i]] = C_group[S[i]] + 1.0
                        else:
                            C_group[S[i]] = C_group[S[i]] + 0.5
                    elif (data_event[i] == False) and (data_event[j] == True) and (prediction[i] < prediction[j]):
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    elif (data_event[i] == True) and (data_event[j] == False) and (prediction[i] > prediction[j]):
                        C_group[S[i]] = C_group[S[i]] + 1.0
                    else:
                        C_group[S[i]] = C_group[S[i]] + 0.5

    CF = C_group / P_group

    CI = 0
    for i in range(len(np.unique(S)) - 1):
        for j in range(i + 1, len(np.unique(S))):
            temp = np.abs(CF[i] - CF[j])
            if temp > CI:
                CI = temp

    return CI * 100


def C_index_difference(unique_elements, S, y_train, event_test, time_test, prediction_test):
    c_index_group = []
    c_td_index_group = []
    for i in unique_elements:
        if len(S.shape) == 1:
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


def CI_td(prediction, data_event, data_time, S):
    CI_list = []
    eval_index = [int(np.percentile(range(prediction.shape[0]), 25)),
                  int(np.percentile(range(prediction.shape[0]), 50)),
                  int(np.percentile(range(prediction.shape[0]), 75))]
    for index in eval_index:
        C_group = np.zeros(len(np.unique(S)), dtype=float)
        P_group = np.zeros(len(np.unique(S)), dtype=float)
        for i in range(len(prediction[0])):
            for j in range(len(prediction[0])):
                if j == i:
                    continue
                else:
                    if ((data_time[i] < data_time[j]) and data_event[i] == False) \
                            or ((data_time[i] > data_time[j]) and data_event[j] == False) \
                            or ((data_time[i] == data_time[j]) and (data_event[i] == False and data_event[j] == False)):
                        continue
                    else:
                        P_group[int(S[i])] = P_group[int(S[i])] + 1.0
                    if data_time[i] < data_time[j]:
                        if prediction[index][i] > prediction[index][j]:
                            C_group[int(S[i])] = C_group[int(S[i])] + 1.0
                        elif prediction[index][i] == prediction[index][j]:
                            C_group[int(S[i])] = C_group[int(S[i])] + 0.5
                    elif data_time[i] > data_time[j]:
                        if prediction[index][i] < prediction[index][j]:
                            C_group[int(S[i])] = C_group[int(S[i])] + 1.0
                        elif prediction[index][i] == prediction[index][j]:
                            C_group[int(S[i])] = C_group[int(S[i])] + 0.5
                    elif data_time[i] == data_time[j]:
                        if data_event[i] == True and data_event[j] == True:
                            if prediction[index][i] == prediction[index][j]:
                                C_group[int(S[i])] = C_group[int(S[i])] + 1.0
                            else:
                                C_group[int(S[i])] = C_group[int(S[i])] + 0.5
                        elif (data_event[i] == False) and (data_event[j] == True) and (
                                prediction[index][i] < prediction[index][j]):
                            C_group[int(S[i])] = C_group[int(S[i])] + 1.0
                        elif (data_event[i] == True) and (data_event[j] == False) and (
                                prediction[index][i] > prediction[index][j]):
                            C_group[int(S[i])] = C_group[int(S[i])] + 1.0
                        else:
                            C_group[int(S[i])] = C_group[int(S[i])] + 0.5

        CF = C_group / P_group

        CI = 0
        for i in range(len(np.unique(S)) - 1):
            for j in range(i + 1, len(np.unique(S))):
                temp = np.abs(CF[i] - CF[j])
                if temp > CI:
                    CI = temp
        CI_list.append(CI)
    CI_score = np.mean(np.asarray(CI_list))
    return CI_score * 100













