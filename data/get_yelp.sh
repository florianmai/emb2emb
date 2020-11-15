

mkdir yelp

# download the data
wget -O yelp/s1.train https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.train.0
wget -O yelp/s2.train https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.train.1
wget -O yelp/s1.dev https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.dev.0
wget -O yelp/s2.dev https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.dev.1
wget -O yelp/s1.test https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.test.0
wget -O yelp/s2.test https://github.com/shentianxiao/language-style-transfer/raw/master/data/yelp/sentiment.test.1

# create file to train AE with
cp yelp/s1.train yelp/all_train
cat yelp/s2.train >> yelp/all_train