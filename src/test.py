#!/usr/bin/python
# -*- coding: utf-8 -*-

#this file is used to load all models specified in the ini file final
#and test it on the devtest set
#run with python test.py ini/test.ini

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
    classifiers = config.get("MODELS", "model_name").split()
    if config.has_option("MODELS", "model_id_nn"):
        model_id_nn= config.get("MODELS", "model_id_nn")
    if config.has_option("MODELS", "model_id_nb"):
        model_id_nb= config.get("MODELS", "model_id_nb")
    if config.has_option("MODELS", "model_id_svm"):
        model_id_svm= config.get("MODELS", "model_id_svm")
    test = config.get("TESTING", "test")

except configparser.NoOptionError:
    raise



#THIS IS WHERE THE FEATURES ARE EXTRACTED AND THE DATA IS PREPARED FOR TRAINING#

if "nb" in classifiers or "baseline" in classifiers:
    print "Reading test data..."
    x_test_nb =SimpleReader(test).prepare_corpus()

#the annotated corpus is send to the FeatureExtractor
#different feature extraction functions are called here
#in order to change which features are used, open feature_extractor.py and change the 
#function calls
if "svm" in classifiers:
    print "Reading training data..."
    test_corpus = CorpusReader(test).prepare_corpus()
   
    print "Extracting features..."
    x_test_svm = FeatureExtractor().extract_features(test_corpus,opinion_pos,opinion_neg,subjectivity_clues)

#the annotated corpus is converted to tensors of pretrained tweet embeddings in order to 
#serve as input to the LSTM 
#NNTrainer takes the following arguments:
#path_to_train_file,path_to_dev_file,embedding_file

if "nn" in classifiers:
    print "Reading training and development data..."
    x_test_nn,y_test_nn,x_dev,y_dev,embedding_layer,max_seq_length,word_index = TensorReader(test,test,pretrained_embeddings).prepare_corpus()


##TRAIN A MODEL##

if "nb" in classifiers:
    nbT= nb.NBTrainer()
    print "Testing..."
    fscore=nbT.test(model_id_nb,x_test_nb)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"
    
elif "nn" in classifiers:
    nnT=nn.NNTrainer()
    print "Testing..."
    fscore=nnT.test(model_id_nn,x_test_nn,y_test_nn)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"
    
elif "svm" in classifiers:
    svmT = svm.SVMTrainer()
    print "Testing..."
    fscore=svmT.test(model_id_svm,x_test_svm)
    print "The model achieves an fscore of "+str(fscore)+" on the development data"

elif "baseline" in classifiers:
    blT=baseline.Baseline()
    print "Testing..."
    fscore=blT.test(x_test_nb)
    print "The baseline achieves an fscore of "+str(fscore)+" on the development data"


