import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data = pd.read_csv(r'train.csv', encoding='unicode_escape')
raw_data.drop(["station"], axis=1, inplace=True)
raw_data[raw_data == 'NR'] = 0
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
row_n = raw_data.shape[0]
theta = np.zeros((9 * 2, 1))                                  
bias = 0
eta = 0.00000001
iteration = 10000
y_hat = pd.DataFrame(raw_data.iloc[range(9, row_n, 18), 11:26])       
y_hat.columns = y_hat.columns.astype(int)
y_hat = y_hat.rename(index={i: i - 9 for i in range(9, row_n, 18)}, columns={i: i - 9 for i in range(9, 24)})
train_data = np.random.choice(np.arange(0, 4320, 18), 192, replace=False)
validation = np.setdiff1d(np.arange(0, 4320, 18), train_data, assume_unique=True)
for it in range(iteration):
    theta_grad = np.zeros((9 * 2, 1))
    b_grad = 0
    Loss = 0
    for row in train_data:                           
        for col in range(2, 17):
            data = raw_data.iloc[[row+8, row+9], col:col+9]
            b_grad = b_grad - 2.0 * (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0])
            theta_grad = theta_grad - 2.0 * (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0]) * data.values.reshape(2*9, 1)
            Loss = Loss + (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0]) ** 2
    bias = bias - eta * b_grad
    theta = theta - eta * theta_grad
    print("Loss:", Loss/2880)
    if Loss/2880 <= 50:
        print("train_finished")
        break
pd.Series(theta[:, 0]).to_csv('model_theta.csv')
pd.Series(bias).to_csv('bias.csv')
result = np.zeros((720, 1))
true_result = np.zeros((720, 1))
i = 0
SSR = 0
SSE = 0
SST = 0
y_sum = 0
for row in validation:
    for col in range(2, 17):
        y_sum += y_hat.loc[row, col-2]
y_aver = y_sum / (validation.size * 15)
for row in validation:
    for col in range(2, 17):
        data = raw_data.iloc[[row+8, row+9], col:col+9]
        result[i] = (data.values.flatten() @ theta) + bias
        true_result[i] = y_hat.loc[row, col-2]
        SSR += (result[i] - y_aver) ** 2
        SSE += (result[i] - true_result[i]) ** 2
        SST += (true_result[i] - y_aver) ** 2
        i += 1
print("SSR", SSR)
print("SSE", SSE)
print("SST", SST)
print("R2", SSR/SST)
plt.plot(np.arange(720), result, label='Prediction')
plt.plot(np.arange(720), true_result, label="Truth")
plt.legend()
plt.show()
