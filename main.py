from cnn import *
from deepNet import *
from one_hidden_layer import *

if __name__ == '__main__':
    print('please input program you want to run:')
    print('1 for one hidden node neural network')
    print('2 for multiple hidden nodes neural network')
    print('3 for convolution neural network')
    count = int(input('input:'))
    if count == 1:
        nn_model()
    elif count == 2:
        multi_layer_model()
    elif count == 3:
        cnn_model()
    else:
        print('input wrong')
