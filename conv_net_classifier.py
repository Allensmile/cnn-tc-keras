import cPickle
import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
import conv_net_dataset
from conv_net_dataset import CnnDataSet

if __name__=="__main__":
    print "args: max_length train-filepath test-filepath dataset-filepath"
    cf = CnnDataSet()
    print "loading training data..."
    max_l = int(sys.argv[1])
    x = cPickle.load(open(sys.argv[2],"rb"))
    revs, cf.W, cf.W2, cf.word_idx_map, cf.vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    trainX, trainY, validateX, validateY = cf.make_cross_data_cv(revs, cf.W, 10, max_l)
    print "loading testing data..."
    x = cPickle.load(open(sys.argv[3],"rb"))
    revs = x[0]
    print "data loaded!"
    testX, testY = cf.make_test_data(revs, cf.W, max_l)


    batch_size = 256
    nb_classes = 2
    nb_epoch = 5

    # input image dimensions
    img_rows, img_cols = cf.nrows, lwn(cf.W[0])
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    nb_pool = 2 
    # convolution kernel size
    kernel_size = (3, 3)

    print '----',len(trainX),' ',trainX.shape[0],' ',img_rows,' ',img_cols
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    validateX = validateX.reshape(validateX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    trainX = trainX.astype('float32')
    validateX = validateX.astype('float32')
    testX = testX.astype('float32')
    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    print(validateX.shape[0], 'validate samples')
    print(testX.shape[0], 'test samples')

    trainY = np_utils.to_categorical(trainY, nb_classes)
    validateY = np_utils.to_categorical(validateY, nb_classes)
    testY = np_utils.to_categorical(testY, nb_classes)


    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax')) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(validateX, validateY))
    score = model.evaluate(testX, testY, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
