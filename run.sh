
python process_data.py data/askprice/vectors.bin data/askprice/posv2.train:1,data/askprice/negv2.train:0 150 train.p

python process_data.py data/askprice/vectors.bin data/askprice/posv2.test:1,data/askprice/negv2.test:0 150 test.p

python conv_net_classifier.py 70 train.p test.p

