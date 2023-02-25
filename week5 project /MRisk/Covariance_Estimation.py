import numpy as np

def covariance_est(df):

    return df.cov()

def exponential_weighted_est(df,lamda):
    asset_num=df.shape[1]
    date_num=df.shape[0]
    weight=np.zeros(date_num)
    for i in range(date_num):
        weight[i] = (1 - lamda) * lamda ** (date_num - i - 1)
    weight = weight / np.sum(weight)

    ew_cov_mat = np.zeros([asset_num, asset_num])
    for i in range(asset_num):
        for j in range(asset_num):
            temp = df.iloc[:, i]
            temp_2 = df.iloc[:, j]
            if i == j:
                ew_cov_mat[i, i] = np.sum(weight * (temp - np.mean(temp)) ** 2)
            else:
                ew_cov_mat[i, j] = np.sum(weight * (temp - np.mean(temp)) * (temp_2 - np.mean(temp_2)))

    return ew_cov_mat

