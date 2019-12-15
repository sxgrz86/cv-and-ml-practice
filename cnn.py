# Read Fashion MNIST dataset
import mnist_reader

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.wrappers import scikit_learn
from keras import optimizers
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# make a network includes 2 convolution, 2 pooling, and 2 dense levels
def creat_conv_model(learning_rate):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=[28,28,1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = keras.optimizers.sgd(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model


def cnn_model():
    # read data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # set parameter and preprocess data
    batchSize = 100
    targetNum = 10
    epoch = 10

    X_train = X_train.reshape(len(X_train), 28, 28, 1)
    X_test = X_test.reshape(len(X_test), 28, 28, 1)
    input_shape = [28, 28, 1]
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255
    X_test /= 255
    y_train = keras.utils.to_categorical(y_train, targetNum)
    y_test = keras.utils.to_categorical(y_test, targetNum)

    # make a dic of hyperparameter and choose best one
    parameter = {'learning_rate': [0.001,0.01,0.1]}
    model1 = scikit_learn.KerasClassifier(build_fn=creat_conv_model)
    grid1 = GridSearchCV(estimator=model1,param_grid=parameter)
    grid_result1 = grid1.fit(X_train,y_train)
    best_lr = grid_result1.best_params_['learning_rate']

    # use best model to train
    best_model = creat_conv_model(best_lr)
    history = best_model.fit(X_train,y_train,
                             batch_size=batchSize,
                             epochs=epoch,
                             validation_split=0.2,
                             shuffle=True)

    # print result of grid search
    print('-'*30)
    print('validation accuracy: {}, best parameter: {}'.format(grid_result1.best_score_,
                                                               grid_result1.best_params_['learning_rate']))
    means = grid_result1.cv_results_['mean_test_score']
    params = grid_result1.cv_results_['params']
    for mean,param in zip(means,params):
        print('{} with {}'.format(mean,param))
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
    print('-'*30)


    # save model
    save_dir = './cnn_model'
    model_name = 'cnn.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir,model_name)
    best_model.save(model_path)
    print('save trained model at %s' % model_path)

    # draw training loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','valid'],loc='upper left')
    plt.savefig('cnn_loss.png')
    plt.show()
