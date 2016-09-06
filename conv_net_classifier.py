import cPickle
import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Embedding
from keras.preprocessing import sequence
import conv_net_dataset
from conv_net_dataset import CnnDataSet
from sklearn import metrics

def build_common_cnn(img_rows, img_cols):
    nb_classes = 2

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2 
    # convolution kernel size
    kernel_size = (2,2)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 1), dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(1, 1), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax')) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model
 
def build_vgg16(img_rows, img_cols):
    nb_classes = 2
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, 1), dim_ordering='tf'))
    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    # build the rest of the network
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
  
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax')) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    return model
 
def build_embedding_cnn(max_caption_len, vocab_size):
    nb_classes = 2
    word_dim = 256

    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    nb_pool = 2 
    # convolution kernel size
    kernel_size = 5
    
    model = Sequential()
    model.add(Embedding(output_dim=word_dim, input_dim=vocab_size, input_length=max_caption_len, name='main_input'))
    #model.add(LSTM(, return_sequences=True))
    model.add(Convolution1D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(Convolution1D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(nb_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax')) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model
 
def run_with_vector():
    print "args: max_length train-filepath test-filepath"
    cf = CnnDataSet()
    print "loading training data..."
    max_l = int(sys.argv[1])
    x = cPickle.load(open(sys.argv[2],"rb"))
    revs, cf.W, cf.W2, cf.word_idx_map, cf.vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"

    U = cf.W2
    head_filter = 2
    cv = 10
    trainX, trainY, validateX, validateY = cf.make_cross_data_cv(revs, U, cv, max_l, head_filter)
    print "loading testing data..."
    x = cPickle.load(open(sys.argv[3],"rb"))
    revs = x[0]
    print "data loaded!"
    testX, testY = cf.make_test_data(revs, U, max_l, head_filter)

    batch_size = 256
    nb_epoch = 10

    # input image dimensions
    img_rows, img_cols = cf.nrows, len(U[0])
    nb_classes = 2

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

    model = build_common_cnn(img_rows, img_cols)  #  build_vgg16(img_rows, img_cols)#
    model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(validateX, validateY))
    score = model.evaluate(testX, testY, verbose=0)
    predY = model.predict_classes(testX, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    confusion = metrics.confusion_matrix(testY[:,1], predY)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print 'Confusion matrix:', confusion
    print 'Accuracy:(TP+TN) / float(TP+TN+FN+FP)=', (TP+TN) / float(TP+TN+FN+FP)
    print 'Error:(FP+FN) / float(TP+TN+FN+FP)=', (FP+FN) / float(TP+TN+FN+FP)
    print 'Recall:TP / float(TP+FN)=',TP / float(TP+FN)
    print 'Specificity:TN / float(TN+FP)=', TN / float(TN+FP)
    print 'False Positive Rate:FP / float(TN+FP)=', FP / float(TN+FP)
    print 'Precision:TP / float(TP+FP)=', TP / float(TP+FP)
    print 'AUC=',metrics.roc_auc_score(testY[:,1], predY)


def run_with_embedding():
    print "args: max_length train-filepath test-filepath dataset-filepath"
    cf = CnnDataSet()
    print "loading training data..."
    max_l = int(sys.argv[1])
    head_filter = 1 
    cv = 5
    x = cPickle.load(open(sys.argv[2],"rb"))
    revs, cf.W, cf.W2, cf.word_idx_map, cf.vocab, cf.idx_word_map = x[0], x[1], x[2], x[3], x[4], x[5]
    print "data loaded!"
    trainX, trainY, validateX, validateY = cf.make_cross_id_data_cv(revs, cv, max_l, head_filter)
    print "loading testing data..."
    x = cPickle.load(open(sys.argv[3],"rb"))
    revs = x[0]
    print "data loaded!"
    testX, testY = cf.make_test_id_data(revs, max_l, head_filter)

    batch_size = 256
    nb_classes = 2
    nb_epoch = 6 
    max_len = 500

    print '---- before ---- trainX[0]:',len(trainX[0]),'trainX.shape:',trainX.shape
    print ' '.join([str(x) for x in trainX[0]])
    print ' '.join(cf.idx2word(trainX[0], cf.idx_word_map))
    
    trainX =  sequence.pad_sequences(trainX, maxlen=max_len)#trainX.reshape(trainX.shape[0],-1,max_len)
    validateX =  sequence.pad_sequences(validateX, maxlen=max_len)#validateX.reshape(validateX.shape[0],-1,max_len)
    testX =  sequence.pad_sequences(testX, maxlen=max_len)#testX.reshape(testX.shape[0],-1,max_len)
    #trainX = trainX.astype('float32')
    #validateX = validateX.astype('float32')
    #testX = testX.astype('float32')
    print '---- after ---- trainX[0]:',len(trainX[0]),'trainX.shape:',trainX.shape
    print ' '.join([str(x) for x in trainX[0]])
    print ' '.join(cf.idx2word(trainX[0], cf.idx_word_map))

    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    print(validateX.shape[0], 'validate samples')
    print(testX.shape[0], 'test samples')

    trainY = np_utils.to_categorical(trainY, nb_classes)
    validateY = np_utils.to_categorical(validateY, nb_classes)
    testY = np_utils.to_categorical(testY, nb_classes)

    model = build_embedding_cnn(max_len, len(cf.vocab))  #  build_vgg16(img_rows, img_cols)#
    model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(validateX, validateY))
    score = model.evaluate(testX, testY, verbose=0)
    predY = model.predict_classes(testX, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    confusion = metrics.confusion_matrix(testY[:,1], predY)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print 'Confusion matrix:', confusion
    print 'Accuracy:(TP+TN) / float(TP+TN+FN+FP)=', (TP+TN) / float(TP+TN+FN+FP)
    print 'Error:(FP+FN) / float(TP+TN+FN+FP)=', (FP+FN) / float(TP+TN+FN+FP)
    print 'Recall:TP / float(TP+FN)=',TP / float(TP+FN)
    print 'Specificity:TN / float(TN+FP)=', TN / float(TN+FP)
    print 'False Positive Rate:FP / float(TN+FP)=', FP / float(TN+FP)
    print 'Precision:TP / float(TP+FP)=', TP / float(TP+FP)
    print 'AUC=',metrics.roc_auc_score(testY[:,1], predY)

if __name__=="__main__":
    run_with_embedding()
