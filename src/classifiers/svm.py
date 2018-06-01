#!/usr/bin/python
# -*- coding: utf-8 -*-

# this class trains a classifier and returns it
from __future__ import unicode_literals
import nltk
import codecs
import time
import datetime
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

###################################################
# Trainer Class using the Support Vector Machine  #
# Algorithm                                       # 
###################################################

class SVMTrainer:
    
    def __init__(self):

        print "SVM Classifier initialized"


    #use the nltk implementation of the Decision Tree algorithm to train 
    def train(self,train_features):
        model = SklearnClassifier(SVC(), sparse=False).train(train_features)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        model_save = open('classifiers/models/svm'+st+'.pickle', 'wb')
        pickle.dump(model, model_save)
        model_save.close()
        return st
        
    def test(self,model_id,dev_features):
        pred=[]
        gold=[]
        model_path = open('classifiers/models/svm'+model_id+".pickle",'rb')
        model = pickle.load(model_path)
        for tweet in dev_features:
            pred.append(model.classify(tweet[0]))
            gold.append(tweet[1])
        return f1_score(gold,pred,average="micro")

