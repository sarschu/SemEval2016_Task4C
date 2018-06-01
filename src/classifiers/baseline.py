#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################
#implements the majority baseline #
#the majority of the tweets are   #
#postitive                        #
###################################


class Baseline:
    def __init__(self):
        print "Initializing the Majority Baseline"
        
    def test(self,dev):
        pred=[]
        gold=[]
        for tweet in dev:
            pred.append("1")
            gold.append(tweet[1])
        return f1_score(gold,pred)
