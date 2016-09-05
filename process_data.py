import numpy as np
import cPickle
from collections import defaultdict
import sys, re, os
import pandas as pd

def read_file(revs, vocab, data_folder, cv=10, clean_string=False):
    sp = data_folder.split(":")
    if len(sp) != 2:
        print "[read_file] file name format error." + data_folder
        return 
    fname = sp[0]
    label = sp[1]
    if not os.path.exists(fname):
            print "[read_file] error file not exists."+  fname
            return 

    with open(fname, "rb") as f:
       for line in f:       
           rev = []
           rev.append(line.strip())
           if clean_string:
               orig_rev = clean_str(" ".join(rev))
           else:
               orig_rev = " ".join(rev).lower()
           words = set(orig_rev.split())
           for word in words:
               vocab[word] += 1
           datum  = {"y":label, 
                     "text": orig_rev,                             
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0,cv)}
           revs.append(datum)

def build_data_cv(data_files, cv=10, clean_string=False):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    for dfile in data_files:
        read_file(revs, vocab, dfile, cv, clean_string)
    return revs, vocab
 
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    layer1_size = 0
    if not os.path.exists(fname):
        print "[load_bin_vec] error file not exists." + fname
        return word_vecs, layer1_size

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
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs, layer1_size

def load_txt_vec(fname, vocab):
    """
    Loads nx1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    layer1_size = 0
    if not os.path.exists(fname):
        print "[load_txt_vec] error file not exists." + fname
        return word_vecs, layer1_size

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
                 word = sp[0]
                 if word in vocab:
                     word_vecs[word] = np.array(map(float, sp[1:]),dtype='float32')  
                     if len(word_vecs[word]) != layer1_size:
                         print "[load_txt_vec] word vector error."
                     else:
                         continue
    return word_vecs, layer1_size

def add_unknown_words(word_vecs, vocab, k=300, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    i = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            i += 1
            if i % 100 ==0:
                print i,' unknown word is added such as ', word,'.'  

def clean_str(string, TREC=False):
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

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":
    if len(sys.argv) != 5:    
        print "args: w2v-filepath data-filepath[file1:0,file2:1,file3:2] dim output-filepath"
        sys.exit(-1)
    w2v_file = sys.argv[1]
    data_files = sys.argv[2]
    data_folder = data_files.split(',')
    k = int(sys.argv[3])
    fpath = sys.argv[4]
    print 'w2v-filepath: ', w2v_file
    print 'data-filepath: ', data_folder
    print 'dim: ', k
    print 'output-filepath: ' + fpath
    print "loading data...\n",        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!\n"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...\n",
    w2v, _ = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!\n"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k)
    W, word_idx_map = get_W(w2v, k)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k)
    W2, _ = get_W(rand_vecs, k)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open(fpath, "wb"))
    print "dataset created!"
