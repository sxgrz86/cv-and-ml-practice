import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read dataset, shuffle it, and extract feature and target

data = pd.read_csv('./wdbc.dataset', header=None)
data = data.sample(frac=1)
y = data.iloc[:,1].copy()
y.loc[y == 'M'] = 1
y.loc[y == 'B'] = 0
x = data.iloc[:,2:].copy()
row_n = x.shape[0]
column_n = x.shape[1]


# 1. partition data into training, testing,
# 2. normalize training data,
# and use scaler of training data to normalize testing data
# 3. transform pd.dataframe into np.ndarray

train_x = x.iloc[0:int(row_n/10*9),:]
test_x = x.iloc[int(row_n/10*9):,:]
train_y = y.iloc[0:int(row_n/10*9)]
test_y = y.iloc[int(row_n/10*9):]

data_max = train_x.max()
data_min = train_x.min()
train_x = (train_x - data_min)/(data_max - data_min)
test_x = (test_x - data_min)/(data_max - data_min)

train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y).reshape(-1,1)
test_y = np.array(test_y).reshape(-1,1)


# 1. define sigmoid function.
# 2. initialize weight and bias, make a set of learning rate.
# 3. losstrack is for recording change of loss from different training process using
# different learning rate.
# Validloss is for recording loss of validation data using different learning rate.
# num_fold is number of fold for cross-validation
# 4. epsilon is a very small constant for avoiding case that log() will
# equal to infinite, which sometimes happens when learning rate is big.

def sigmoid(z):
    return 1/(1 + np.exp(-z))


num_fold = 9
valid_num = int(len(train_x)/num_fold)
losstrack = []
accuracytrack = []
learningrate = [0.01, 0.05, 0.1, 0.5, 1, 10]
validloss = []
hyperparameter_n = 5
w = np.ones([column_n, hyperparameter_n]).reshape(-1,hyperparameter_n)
b = np.zeros(hyperparameter_n).reshape(-1,1)
epsilon = 1e-5

# train data using logistics regression model
# use 9 fold cross validation to train
# choose best learning rate according to mean of validation error
# some temperate variables are created to save weight, bias, loss etc.
for i in range(hyperparameter_n):
    w_temp = np.ones([column_n, num_fold])
    b_temp = np.zeros(num_fold).reshape(-1,1)
    losstrack_temp = []
    validloss_temp = []
    accuracy_temp = []
    for fold in range(num_fold):
        losstrack_temp.append([])
        accuracy_temp.append([])
        tw = np.ones(column_n).reshape(-1, 1)
        valid_data = train_x[valid_num * fold:valid_num * (fold+1),:]
        train_data = np.concatenate((train_x[:valid_num * fold,:], train_x[valid_num * (fold+1):,:]))
        valid_target = train_y[valid_num * fold:valid_num * (fold+1)]
        train_target = np.concatenate((train_y[:valid_num * fold,:], train_y[valid_num * (fold+1):,:]))

        for j in range(5000):
            h = sigmoid(np.dot(train_data, tw) + b_temp[fold])
            loss = -np.sum((np.log(h + epsilon) * train_target + (1 - train_target) * np.log(1 - h + epsilon)))/train_target.shape[0]
            losstrack_temp[fold].append(loss)
            result = np.rint(h)
            acc = 1 - np.sum(np.abs(result - train_target))/len(result)
            accuracy_temp[fold].append(acc)
            dw = np.dot(train_data.T, (h - train_target))/train_target.shape[0]
            db = np.sum(h - train_target)/train_target.shape[0]

            tw = tw - learningrate[i] * dw
            b_temp[fold] = b_temp[fold] - learningrate[i] * db

        for k in range(len(tw)):
            w_temp[k,fold] = tw[k]
        v_y = sigmoid(np.dot(valid_data, tw) + b_temp[fold])
        l_v = -np.sum(valid_target * np.log(v_y + epsilon) + (1 - valid_target) * np.log(1 - v_y + epsilon)) / valid_target.shape[0]
        validloss_temp.append(l_v)

    minindex = int(np.argmin(validloss_temp))
    for k in range(len(tw)):
        w[k,i] = w_temp[k,minindex]
        b[i] = b_temp[minindex]
    losstrack.append(losstrack_temp[minindex])
    validloss.append(np.mean(validloss_temp))
    accuracytrack.append(accuracy_temp[minindex])
    print(learningrate[i])


# choose best learning rate and use it to test data
minindex = int(np.argmin(validloss))
print('best learning rate:%.3f' % (learningrate[minindex]))
tw = w[:,minindex].reshape(-1,1)
t_y = sigmoid(np.dot(test_x, tw) + b[minindex])
test_loss = -np.sum(test_y * np.log(t_y + epsilon) + (1 - test_y) * np.log(1 - t_y + epsilon)) / test_y.shape[0]
plt.subplot(211), plt.plot(losstrack[minindex])
plt.title('losstrack')
plt.grid()
plt.subplot(212), plt.plot(accuracytrack[minindex])
plt.title('accuracytrack')
plt.grid()
print('test loss:%f' % test_loss)


# compare predict result with target, and calculate confusion matrix
result = np.rint(t_y)
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(result)):
    if((test_y[i]==1)&(result[i]==1)):
        tp += 1
    elif((test_y[i]==1)&(result[i]==0)):
        fn += 1
    elif ((test_y[i] == 0) & (result[i] == 1)):
        fp += 1
    elif ((test_y[i] == 0) & (result[i] == 0)):
        tn += 1
accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print('accuracy: %f' % accuracy)
print('precision: %f' % precision)
print('recall: %f' % recall)
plt.show()







