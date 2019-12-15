# Read Fashion MNIST dataset
import mnist_reader

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.wrappers import scikit_learn
from keras import optimizers
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def creat_model(learning_rate,hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes,input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dense(hidden_nodes))
    model.add(Activation('sigmoid'))
    model.add(Dense(hidden_nodes))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = optimizers.sgd(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model


'''
# I create 3 model
# it takes too long to tune 2 parameter at once
# model 1 is for tuning learning rate
# model 2 is for tuning hidden nodes
# best model uses the best hyperparameter to train model
# training loss graph is from this best model
# then it is used on test set, and get final test accuracy, confusion matrix
'''


def multi_layer_model():
    # read data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # process data
    batchSize = 100
    epoch = 10
    targetNum = 10
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255
    X_test /= 255
    y_train = keras.utils.to_categorical(y_train, targetNum)
    y_test = keras.utils.to_categorical(y_test, targetNum)

    # make a dict of hyperparameter and chose the best
    parameter1 = {'learning_rate': [0.001,0.01,0.1],
                  'hidden_nodes': [100]}
    model1 = scikit_learn.KerasClassifier(build_fn=creat_model)
    grid1 = GridSearchCV(estimator=model1,param_grid=parameter1)
    grid_result1 = grid1.fit(X_train,y_train)
    best_lr = grid_result1.best_params_['learning_rate']

    parameter2 = {'learning_rate': [best_lr],
                  'hidden_nodes': [100,200,300]}
    model2 = scikit_learn.KerasClassifier(build_fn=creat_model)
    grid2 = GridSearchCV(estimator=model2,param_grid=parameter2)
    grid_result2 = grid2.fit(X_train,y_train)
    best_nodes = grid_result2.best_params_['hidden_nodes']

    # use best model to train again
    best_model = creat_model(best_lr,best_nodes)
    history = best_model.fit(X_train,y_train,
                             batch_size=batchSize,
                             epochs=epoch,
                             validation_split=0.2,
                             shuffle=True)

    # print result of grid search
    print('-'*30)
    print('accurary: {}, best parameter: {}'.format(grid_result1.best_score_, grid_result1.best_params_['learning_rate']))
    means = grid_result1.cv_results_['mean_test_score']
    params = grid_result1.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean,param))
    print('-'*30)

    print('accurary: {}, best parameter: {}'.format(grid_result2.best_score_, grid_result2.best_params_['hidden_nodes']))
    means = grid_result2.cv_results_['mean_test_score']
    params = grid_result2.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))
    print('-'*30)

    # print test accuracy
    loss,accuracy = best_model.evaluate(X_test,y_test)
    print('test set accuracy: {}'.format(accuracy))
    print('-'*30)

    # make confusion matrix
    y_pre = best_model.predict(X_test)
    y_pred = np.argmax(y_pre,axis=1)
    y_test = np.argmax(y_test,axis=1)
    print('confusion matrix')
    print(confusion_matrix(y_test,y_pred))


    # save model
    save_dir = './dnn_model'
    model_name = 'dnn.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir,model_name)
    best_model.save(model_path)
    print('save trained model at %s' % model_path)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','valid'],loc='upper left')
    plt.savefig('dnn_loss.png')
    plt.show()

