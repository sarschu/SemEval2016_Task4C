#!/usr/bin/python
# -*- coding: utf-8 -*-

# this class trains a classifier and returns it
from __future__ import unicode_literals
import nltk
import codecs
import time
import datetime
from nltk.classify import NaiveBayesClassifier
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from sklearn.metrics import f1_score
import pickle

###################################################
# Trainer Class using the Naive Bayes Algorithm   #
###################################################
        
class NBTrainer:

    def __init__(self):
        #train the classifier right at its initialization
        print("Naive Bayes Classifier initialized")


     #use the nltk implementation of the Naive Bayes algorithm to train
    def train(self,train):
    
        model = NaiveBayesClassifier(train)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        model_save = open('classifiers/models/naive_bayes'+st+'.pickle', 'wb')
        pickle.dump(model, model_save)
        model_save.close()
        return st

    def test(self,model_id,test):
        pred=[]
        gold=[]
        model_path = open('classifiers/models/naive_bayes'+model_id+".pickle",'rb')
        model = pickle.load(model_path)

        for tweet in test:

            pred.append(model.classify(tweet[0]))
            gold.append(tweet[1])
        return f1_score(gold,pred,average="micro")



        
