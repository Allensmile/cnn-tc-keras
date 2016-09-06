# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd

class CnnDataSet:

    ncols = 200
    nrows = 59
    W = []
    W2 = []
    word_idx_map = dict()
    idx_word_map = dict()
    vocab = defaultdict(float)

    def __init__(self):
        pass

    def get_idx_from_sent(self, sent, max_l=51, filter_h=5, expanded=True):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = sent.split()
        for word in words:
            if word in self.word_idx_map:
                if expanded == True and len(x) < max_l+2*pad:
                    x.append(self.word_idx_map[word])
                if expanded == False:
                    x.append(self.word_idx_map[word])

        if expanded == True:
            while len(x) < max_l+2*pad:
                x.append(0)
            self.nrows = max_l+2*pad
        else:
            for i in xrange(pad):
                x.append(0)
            self.nrows = len(x)

        return x

    def search(self, sent, w):
        x = []
        for word in sent:
            if len(w) < word:
                print "[CnnDataSet] word id error", word, " total length ", len(w)
                break
            x.append(w[word])
        return np.array(x)

    def make_cross_data_cv(self, revs, w, cv, max_l=51, filter_h=5):
        """
        Transforms sentences into a 2-d matrix.
        """
        trainX, trainY, testX, testY  = [], [], [], []
        for rev in revs:        
            sent = self.get_idx_from_sent(rev["text"], max_l, filter_h)
            sample = self.search(sent, w) 
            if rev["split"]==cv:            
                testX.append(sample)        
                testY.append(rev["y"])
            else:  
                trainX.append(sample)   
                trainY.append(rev["y"])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        return [trainX, trainY, testX, testY]     

    def make_cross_id_data_cv(self, revs, cv, max_l=51, filter_h=5, expended=False):
        """
        Transforms sentences into a 2-d matrix.
        """
        trainX, trainY, testX, testY  = [], [], [], []
        for rev in revs:        
            sent = self.get_idx_from_sent(rev["text"], max_l, filter_h, expended)
            if rev["split"]==cv:            
                testX.append(sent)        
                testY.append(rev["y"])
            else:  
                trainX.append(sent)   
                trainY.append(rev["y"])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        return [trainX, trainY, testX, testY]     


    def make_test_data(self, revs, w, max_l=51, filter_h=5):
        """
        Transforms sentences into a 2-d matrix.
        """
        testX, testY  = [], []
        for rev in revs:           
            sent = self.get_idx_from_sent(rev["text"], max_l, filter_h)
            sample = self.search(sent, w)
            testX.append(sample)   
            testY.append(rev["y"])
        testX = np.array(testX)
        testY = np.array(testY)
        return [testX, testY]

    def make_test_id_data(self, revs, w, max_l=51, filter_h=5):
        """
        Transforms sentences into a 2-d matrix.
        """
        testX, testY  = [], []
        for rev in revs:           
            sent = self.get_idx_from_sent(rev["text"], max_l, filter_h, False)
            testX.append(sent)   
            testY.append(rev["y"])
        testX = np.array(testX)
        testY = np.array(testY)
        return [testX, testY]

    def idx2word(self, words, idx_word_map):
        """
        Transforms sentences into a 2-d matrix.
        """
        x = []
        for word in words:
            if idx_word_map.has_key(word):
                x.append(idx_word_map[word])
            else:
                x.append('_')
        return x
     
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
    cPickle.dump([trainX, trainY, validateX, validateY, testX, testY], open(sys.argv[4], "wb"))
