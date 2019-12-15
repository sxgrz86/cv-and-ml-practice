# Read Fashion MNIST dataset
import mnist_reader

# Your code goes here . . .
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    for i in range(len(z)):
        for j in range(len(z[0])):
            z[i,j] = 1/(1 + np.exp(-z[i,j]))
    return z


def softmax(z):
    for i in range(len(z)):
        expsum = 0
        for j in range(len(z[0])):
            z[i,j] = np.exp(z[i,j])
            expsum += z[i,j]
        for j in range(len(z[0])):
            z[i,j] /= expsum
    return z


def nn_model():
    # read data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # normalize data
    data_train = X_train/255
    data_test = X_test/255

    # partition data
    sampleNum = int(len(data_train)/10)
    data_valid = data_train[9*sampleNum:,:]
    data_train = data_train[:9*sampleNum,:]

    # make a target matrix where each element represent probability
    target0 = np.zeros((len(y_train),10),dtype=int)
    for i in range(len(y_train)):
        cla = y_train[i]
        target0[i,cla] = 1

    target_train = target0[:9*sampleNum,:]
    target_valid = target0[9*sampleNum:,:]

    target_test = np.zeros((len(y_test),10),dtype=int)
    for i in range(len(y_test)):
        cla = y_test[i]
        target_test[i,cla] = 1

    # set parameter
    hiddenNodes = [50,100,300]
    featureNum = data_train.shape[1]
    targetNum = 10
    batchSize = 100
    batchNum = int(data_train.shape[0]/batchSize)
    epsilon = 1e-5
    learningRate = [0.001,0.01,0.1]
    epoch = 10
    lamb = 0.001


    # these list saves parameter of models with different hyper parameter
    lossList1 = []
    lossList2 = []
    trackList = []
    wList = []
    wbList = []
    vList = []
    vbList = []

    for lr in range(len(learningRate)):
        v = np.zeros((featureNum, hiddenNodes[0]))
        vb = np.zeros((1, hiddenNodes[0]))
        w = np.zeros((hiddenNodes[0], targetNum))
        wb = np.zeros((1, targetNum))

        for i in range(epoch):
            for j in range(batchNum):
                print("hidden nodes:{},learning rate:{},  epoch:{}/{},batch:{}/{}".format(
                    hiddenNodes[0],learningRate[lr],i+1,epoch,(j+1)*batchSize,9*sampleNum))
                data = data_train[j*batchSize:(j+1)*batchSize]
                t = target_train[j*batchSize:(j+1)*batchSize]

                a1 = sigmoid(np.dot(data,v) + vb)
                a2 = softmax(np.dot(a1,w) + wb)

                d2 = (a2-t)
                d1 = (np.multiply(np.dot(d2,w.T),np.multiply(a1,(1 - a1))))

                dw = np.dot(a1.T, d2)
                dwb = np.sum(d2,axis=0)
                w = w - learningRate[lr] * (dw + lamb * w)
                wb = wb - learningRate[lr] * dwb

                dv = np.dot(data.T, d1)
                dvb = np.sum(d1,axis=0)
                v = v - learningRate[lr] * (dv + lamb * v)
                vb = vb - learningRate[lr] * dvb

        # test parameter on validation set
        a1 = sigmoid(np.dot(data_valid,v) + vb)
        a2 = softmax(np.dot(a1,w) + wb)
        valid_loss = (-np.sum(np.multiply(np.log(a2 + epsilon),target_valid)))
        lossList1.append(valid_loss)

    bestmodel = np.argmin(lossList1)
    best_lr_rate = learningRate[bestmodel]

    for nd in range(len(hiddenNodes)):
        losstrack = []
        v = np.zeros((featureNum, hiddenNodes[nd]))
        vb = np.zeros((1, hiddenNodes[nd]))
        w = np.zeros((hiddenNodes[nd], targetNum))
        wb = np.zeros((1, targetNum))

        for i in range(epoch):
            for j in range(batchNum):
                print("hidden nodes:{},learning rate:{}, epoch:{}/{},batch:{}/{}".format(
                    hiddenNodes[nd],best_lr_rate,i+1,epoch,(j+1)*batchSize,9*sampleNum))
                data = data_train[j*batchSize:(j+1)*batchSize]
                t = target_train[j*batchSize:(j+1)*batchSize]

                a1 = sigmoid(np.dot(data,v) + vb)
                a2 = softmax(np.dot(a1,w) + wb)
                loss = -np.sum(np.multiply(np.log(a2 + epsilon),t))
                losstrack.append(loss)

                d2 = (a2-t)
                d1 = (np.multiply(np.dot(d2,w.T),np.multiply(a1,(1 - a1))))

                dw = np.dot(a1.T, d2)
                dwb = np.sum(d2,axis=0)
                w = w - best_lr_rate * (dw + lamb * w)
                wb = wb - best_lr_rate * dwb

                dv = np.dot(data.T, d1)
                dvb = np.sum(d1,axis=0)
                v = v - best_lr_rate * (dv + lamb * v)
                vb = vb - best_lr_rate * dvb

        a1 = sigmoid(np.dot(data_valid,v) + vb)
        a2 = softmax(np.dot(a1,w) + wb)
        valid_loss = (-np.sum(np.multiply(np.log(a2 + epsilon),target_valid)))
        lossList2.append(valid_loss)
        trackList.append(losstrack)
        wList.append(w)
        wbList.append(wb)
        vList.append(v)
        vbList.append(vb)

    bestmodel = np.argmin(lossList2)
    bestnode = hiddenNodes[bestmodel]

    # get weight and bias from best model
    w = wList[bestmodel]
    wb = wbList[bestmodel]
    v = vList[bestmodel]
    vb = vbList[bestmodel]
    losstrack = trackList[bestmodel]

    # print validation result
    print('best learning rate %f' % best_lr_rate)
    print('best hidden nodes: {}'.format(bestnode))
    for i in range(len(lossList1)):
        print('learning rate: {}, validation loss: {}'.format(learningRate[i],lossList1[i]))

    for i in range(len(lossList2)):
        print('hidden nodes: {}, validation loss: {}'.format(hiddenNodes[i],lossList2[i]))
    print('-'*30)

    # test model on testing set and print test accuracy
    a1 = sigmoid(np.dot(data_test,v)+vb)
    a2 = softmax(np.dot(a1,w)+wb)
    predict_result = np.zeros((target_test.shape[0],targetNum),dtype=int)
    for i in range(len(target_test)):
        maxindex = np.argmax(a2[i])
        predict_result[i,maxindex] = 1

    accuracy = 1 - np.sum(np.abs(predict_result - target_test))/2/len(data_test)
    print('test accuracy: {}'.format(accuracy))
    print('-'*30)

    print('test accuracy %f' % accuracy)
    print('learning rate: {}, hidden nodes: {}'.format(best_lr_rate,bestnode))

    # make confusion matrix
    # matrix1 saves tp,fp,fn,tn
    # matrix2 saves correct and wrong classification result
    y_t = np.argmax(target_test,axis=1)
    y_pre = np.argmax(predict_result,axis=1)
    confusion_matrix1 = []
    confusion_matrix2 = np.zeros((targetNum,targetNum),dtype=int)
    print('confusion matrix1')
    for i in range(targetNum):
        tp = 0; tn = 0; fp = 0; fn = 0
        for j in range(len(target_test)):
            if (y_pre[j] == i) & (y_t[j] == i):
                tp += 1
                confusion_matrix2[i][i] += 1
            elif (y_pre[j] == i) & (y_t[j] != i):
                fp += 1
            elif (y_pre[j] != i) & (y_t[j] == i):
                fn += 1
                confusion_matrix2[i][y_pre[j]] += 1
            elif (y_pre[j] != i) & (y_t[j] != i):
                tn += 1

        print('class {},{}'.format(i,[tp,fp,fn,tn]))
        confusion_matrix1.append([tp,fp,fn,tn])

    table = []
    for i in range(len(table)):
        precision = float(confusion_matrix1[i][0])/(confusion_matrix1[i][0] + confusion_matrix1[i][1])
        recall = float(confusion_matrix1[i][0])/(confusion_matrix1[i][0] + confusion_matrix1[i][2])
        table.append([precision,recall])

    print('-'*30)
    print('precision and recall')
    print(table)
    print('_'*30)
    print('confusion matrix2')
    print(confusion_matrix2)
    print('_'*30)
    print(np.sum(confusion_matrix2))

    # draw plot of training loss
    plt.figure()
    plt.plot(losstrack)
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.savefig('one_hidden_loss2.png')
    plt.show()
