import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

class DataPrepare:

    vecdim = 200
    revs = []
    W = []
    W2 = []
    word_idx_map = dict()
    vocab = defaultdict(float)

    def __init__(self):
        pass

    def read_file(self,data_folder, cv=10, clean_string=True):
        sp = data_folder.split(":")
        if len(sp) != 2:
            print "[read_file] file name format error."
            return 
        with open(sp[0], "rb") as f:
           for line in f:       
               rev = []
               rev.append(line.strip())
               if clean_string:
                   orig_rev = clean_str(" ".join(rev))
               else:
                   orig_rev = " ".join(rev).lower()
               words = set(orig_rev.split())
               for word in words:
                   self.vocab[word] += 1
               datum  = {"y":sp[1], 
                         "text": orig_rev,                             
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0,cv)}
               self.revs.append(datum)
        
    def build_data_cv(self, data_files, cv=10, clean_string=True):
        """
        Loads data and split into 10 folds. data_folder=["file1:0","file2:1","file3:2"]
        """
        pos_file = data_files[0]
        neg_file = data_files[1]
        self.read_file(pos_file, cv, clean_string)
        self.read_file(neg_file, cv, clean_string)
    
    def get_W(self, word_vecs, freed=0):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        if freed == 0:
            freed = self.vecdim
        vocab_size = len(word_vecs)
        self.W = np.zeros(shape=(vocab_size+1, self.vecdim), dtype='float32')            
        self.W2 = np.zeros(shape=(vocab_size+1, freed), dtype='float32')            
        self.W[0] = np.zeros(self.vecdim, dtype='float32')
        self.W2[0] = np.zeros(freed, dtype='float32')
        i = 1
        for word in word_vecs:
            self.W[i] = word_vecs[word]
            self.W2[i] = np.random.uniform(-0.25,0.25,freed)
            self.word_idx_map[word] = i
            i += 1
    
    def load_bin_vec(self, fname):
        """
        Loads nx1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
        self.vecdim = layer1_size
        return word_vecs

    def load_txt_vec(self, fname):
        """
        Loads nx1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            word = []
            while True:
                lines = f.readlines(100000)
                if not lines:
                     break    
                for line in lines:
                     sp = line.split()
                     word_vecs[sp[0]] = np.array(map(float, sp[1:]),dtype='float32')  
                     if len(word_vecs[sp[0]]) != layer1_size:
                         print "[load_txt_vec] word vector error."
        self.vecdim = layer1_size
        return word_vecs


    def add_unknown_words(self, word_vecs, min_df=1):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """ 
        for word in self.vocab:
            if word not in word_vecs or self.vocab[word] < min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,self.vecdim)  

    def clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip() if TREC else string.strip().lower()

    def clean_str_sst(self, string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()

if __name__=="__main__":   
    print "args: w2v-filepath positive-filepath negative-filepath output-filepath dim" 
    w2v_file = sys.argv[1]     
    data_files = [sys.argv[2]+":1",sys.argv[3]+":0"]
    print "loading data..."
    dp = DataPrepare()
    dp.build_data_cv(data_files, 10, False)
    max_l = np.max(pd.DataFrame(dp.revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(dp.revs))
    print "vocab size: " + str(len(dp.vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = dp.load_bin_vec(w2v_file)
    print "word2vec loaded!"
    print "num words already in word2vec: " , len(w2v)
    dp.add_unknown_words(w2v)
    dp.get_W(w2v, int(sys.argv[5]))
    cPickle.dump([dp.revs, dp.W, dp.W2, dp.word_idx_map, dp.vocab], open(sys.argv[4], "wb"))
    print "dataset created!"
    
