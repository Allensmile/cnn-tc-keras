#python process_data.py True data/askprice/vectors.bin data/askprice/positive.train data/askprice/negative.train train.p 100
#python process_data.py True data/askprice/vectors.bin data/askprice/positive.test data/askprice/negative.test test.p 100

python conv_net_classifier.py 51 train.p test.p

