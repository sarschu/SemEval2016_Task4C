#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################
#implements the majority baseline #
#the majority of the tweets are   #
#postitive                        #
###################################

from sklearn.metrics import f1_score

class Baseline:
    def __init__(self):
        print "Initializing the Majority Baseline"
        
    def test(self,test):
        pred=[]
        gold=[]
        for tweet in test:
            pred.append("1")
            gold.append(tweet[1])
        return f1_score(gold,pred,average="micro")
