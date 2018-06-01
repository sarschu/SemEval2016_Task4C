#!/usr/bin/python
# -*- coding: utf-8 -*-
# The code has been adapted from https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa

import numpy as np
import codecs
import csv
import time
import datetime
from keras.models import Sequential
from keras import metrics
from keras.layers import LSTM, Embedding,Input,Dropout, Conv1D, MaxPooling1D,Dense
from keras import optimizers
from keras.callbacks import Callback
from sklearn.metrics import f1_score
import keras.backend as K

###################################################
# Trainer Class using a Recurrent Neural Net      #
# Algorithm (LSTM)                                #
###################################################

class NNTrainer:
    def __init__(self):
        print("Reccurent Neural Net classifier initialized")
        #self.metrics = Metrics()
        
    #keras does not offer f1 score by default   
    def _precision(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def _recall(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
        
    def fbeta_score(self,y_true, y_pred, beta=1):
        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = self._precision(y_true, y_pred)
        r = self._recall(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

                            
    def train(self,x_train,y_train,x_val,y_val,embedding_matrix,MAX_SEQUENCE_LENGTH,word_index):
        model = Sequential()
        vocabulary_size=embedding_matrix.shape[0]
        model.add(Embedding(vocabulary_size, 100, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(LSTM(100))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[self.fbeta_score])
        ## Fit train data
        model.fit(x_train, np.array(y_train), validation_data=(x_val,y_val), epochs = 3)
        
        # serialize model to JSON
        model_json = model.to_json()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        with open("classifiers/models/nn"+st+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("classifiers/models/nn"+st+".h5")
        print("Saved model to disk")    
        return st

    def test(self,model_id,X,Y):
        # load json and create model
        json_file = open('classifiers/models/nn'+model_id+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("classifiers/models/nn"+model_id+".h5")
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[self.fbeta_score])
        score = loaded_model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        return score

 
