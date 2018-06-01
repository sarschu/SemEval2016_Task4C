#!/usr/bin/python
# encoding: utf8
# -*- coding: utf-8 -*-

#call python util.py corpus_data

import string
import sys
from data_reader import CorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist

#get insights into data
def show_label_distribution(data):
    very_neg,neg,neu,pos,very_pos=0,0,0,0,0

    for m in data:
        message = data[m]

        if message["label"]=="-2":
            very_neg += 1
        elif message["label"]=="-1":
            neg += 1
        elif message["label"]=="0":
            neu += 1
        elif message["label"]=="1":
            pos += 1
        elif message["label"]=="2":
            very_pos += 1
            
    print("Number of very negative Tweets")
    print(very_neg)
    print("Number of  negative Tweets")
    print(neg)
    print("Number of neutral Tweets")
    print(neu)
    print("Number of positive Tweets")
    print(pos)
    print("Number of very positive Tweets")
    print(very_pos)

def mfw_per_label(data):
    very_neg,neg,neu,pos,very_pos=[],[],[],[],[]
    #data_reader = DataReader(corpus_data)
    #remove stop words
    stop_words = set(stopwords.words('english'))
    for m in data:
        message = data[m]
        if message["label"]=="-2":
            very_neg += [x for x in message["lemmatized_tweet"] if x not in stop_words]
        elif message["label"]=="-1":
            neg += [x for x in message["lemmatized_tweet"] if x not in stop_words]
        elif message["label"]=="0":
            neu += [x  for x in message["lemmatized_tweet"] if x not in stop_words]
        elif message["label"]=="1":
            pos += [x  for x in message["lemmatized_tweet"] if x not in stop_words]
        elif message["label"]=="2":
            very_pos += [x  for x in message["lemmatized_tweet"] if x not in stop_words]


    fdist_very_neg = FreqDist(very_neg)
    fdist_neg = FreqDist(neg)
    fdist_neu = FreqDist(neu)
    fdist_pos = FreqDist(pos)
    fdist_very_pos = FreqDist(very_pos)

    #print the most frequent words for each label
    print("50 most frequent words in very negative tweets:"
    print(fdist_very_neg.most_common(50))
    print("50 most frequent words in  negative tweets:"
    print(fdist_neg.most_common(50))
    print("50 most frequent words in neutral tweets:"
    print(fdist_neu.most_common(50))
    print("50 most frequent words in positive tweets:"
    print(fdist_pos.most_common(50))
    print("50 most frequent words in very positive tweets:"
    print(fdist_very_pos.most_common(50))
     
        
def main():
    corpus_data = CorpusReader(sys.argv[1]).prepare_corpus()
    show_label_distribution(corpus_data)
    mfw_per_label(corpus_data)
        
if __name__ == "__main__":
    main()
