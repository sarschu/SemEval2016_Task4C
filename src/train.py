#!/usr/bin/python
# -*- coding: utf-8 -*-

# this file is is the call script to train and test a model
# call with: python train.py ini/train_all.ini
import configparser
from data_reader import CorpusReader,TensorReader, SimpleReader
from feature_extractor import FeatureExtractor
from classifiers import *
import os
import sys


#THIS IS WHERE THE INPUT FILES ARE DEFINED#
config = configparser.ConfigParser()
config.read(unicode(sys.argv[1]))
try:
    if config.has_option('DICTIONARIES', 'opinion_neg'):
        opinion_neg = config.get("DICTIONARIES", "opinion_neg")
    if config.has_option('DICTIONARIES', 'opinion_pos'):
        opinion_pos = config.get("DICTIONARIES", "opinion_pos")
    if config.has_option('DICTIONARIES', 'subjectivity_clues'):
        subjectivity_clues=config.get("DICTIONARIES", "subjectivity_clues")
    if config.has_option("TRAINING", "embedding_file"):
        pretrained_embeddings=config.get("TRAINING", "embedding_file")
    classifier = config.get("TRAINING", "trainer")
    
    train = config.get("TRAINING", "train")
    dev=config.get("TRAINING", "dev")

except configparser.NoOptionError:
    raise




#THIS IS WHERE THE FEATURES ARE EXTRACTED AND THE DATA IS PREPARED FOR TRAINING#

if classifier =="nb" or classifier =="baseline":
    print "Reading training data..."
    x_train = SimpleReader(train).prepare_corpus()
    print "Reading development data..."
    x_dev =SimpleReader(dev).prepare_corpus()

#the annotated corpus is send to the FeatureExtractor
#different feature extraction functions are called here
#in order to change which features are used, open feature_extractor.py and change the 
#function calls
if classifier =="svm":
    print "Reading training data..."
    train_corpus = CorpusReader(train).prepare_corpus()
    print "Reading development data..."
    dev_corpus = CorpusReader(dev).prepare_corpus()
    print "Extracting features..."
    x_train = FeatureExtractor().extract_features(train_corpus,opinion_pos,opinion_neg,subjectivity_clues)
    x_dev =FeatureExtractor().extract_features(dev_corpus,opinion_pos,opinion_neg,subjectivity_clues)

#the annotated corpus is converted to tensors of pretrained tweet embeddings in order to 
#serve as input to the LSTM 
#NNTrainer takes the following arguments:
#path_to_train_file,path_to_dev_file,embedding_file

if classifier =="nn":
    print "Reading training and development data..."
    x_train,y_train,x_dev,y_dev,embedding_layer,max_seq_length,word_index = TensorReader(train,dev,pretrained_embeddings).prepare_corpus()


##TRAIN A MODEL##

if classifier =="nb":
    nbT= nb.NBTrainer()
    print "Training..."
    model_id = nbT.train(x_train)
    print "Testing..."
    fscore=nbT.test(model_id,x_dev)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"
    
elif classifier =="nn":
    nnT=nn.NNTrainer()
    print "Training..."
    model_id = nnT.train(x_train,y_train,x_dev,y_dev,embedding_layer,max_seq_length,word_index)
    print "Testing..."
    fscore=nnT.test(model_id,x_dev,y_dev)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"
    
elif classifier =="svm":
    svmT = svm.SVMTrainer()
    print "Training..."
    model_id = svmT.train(x_train)
    print "Testing..."
    fscore=svmT.test(model_id,x_dev)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"

elif classifier =="baseline":
    blT=baseline.Baseline()
    print "Testing..."
    fscore=blT.test(x_dev)
    print "The baseline achieves an fscore of "+str(fscore)+" on the development data"


