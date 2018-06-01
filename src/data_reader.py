#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk.tokenize import TweetTokenizer
import codecs
import csv
import re
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import *


"""This DataReader reads the file format provided by the semeval shared task 
    and returns a python dictionary containing relevant information for the feature extractor"""
class CorpusReader:
    
    # this is the constructor of the CorpusReader class. It makes the
    # corpus file available to the class
    def __init__(self, path_to_file):
        tsvfile = codecs.open(path_to_file)
        self.corpus_data = csv.reader(tsvfile, delimiter='\t')
        self.corpus = {}

  
    def prepare_corpus(self):

        for row in self.corpus_data:
            # add the data available in the tsv file to the corpus dictionary 
            #replace the links with a placeholder (http) to avoid encoding problems
            #if text says "Not available" skip the tweet
            try:
                self.corpus[int(row[0])]={"topic":row[1],"label":row[2],"original_tweet":unicode(row[3]).encode("utf8"),"lemmatized_tweet":[]}
                      
                #################
                # PREPROCESSING #
                #################
           
                # add the tokenized text using a dedicated Tweet tokenizer
                tweet_tokenizer = TweetTokenizer()
                self.corpus[int(row[0])]["tokenized_tweet"] = tweet_tokenizer.tokenize(self.corpus[int(row[0])]["original_tweet"])

                # add the lemmatized text
                lmt = WordNetLemmatizer()
                for word in self.corpus[int(row[0])]["tokenized_tweet"]:
                    self.corpus[int(row[0])]["lemmatized_tweet"].append(lmt.lemmatize(word))
                
                #########################
                # LINGUISTIC ENRICHMENT #
                #########################
                
                # add part-of-speech 
                self.corpus[int(row[0])]["pos"] = nltk.pos_tag(self.corpus[int(row[0])]["tokenized_tweet"])

                #add chunking
                self.corpus[int(row[0])]["chunked"] =nltk.chunk.ne_chunk(self.corpus[int(row[0])]["pos"])

            except:
                print( "the following message contains encoding problems")
                print(row )
        return self.corpus
              
class TensorReader:

    def __init__(self,path_to_train_file,path_to_dev_file,embedding_file):
        tsvtrain = codecs.open(path_to_train_file)
        self.corpus_data_train = csv.reader(tsvtrain, delimiter='\t')
        tsvdev = codecs.open(path_to_dev_file)
        self.corpus_data_dev = csv.reader(tsvdev, delimiter='\t')
        self.embeddings=open(embedding_file)
        
    def prepare_corpus(self):

        # dictionary mapping label name to numeric id
        labels_index = {"-2":-2,"-1":-1,"0":0,"1":1,"2":2}  
        MAX_NB_WORDS=0
        #read train data
        texts_train = []  # list of text samples
        tweet_tokenizer = TweetTokenizer()
        labels_train = []  # list of label ids
        for row in self.corpus_data_train:
            ## Convert words to lower case and split them
            text = row[3].lower()
            text = tweet_tokenizer.tokenize(unicode(row[3]).encode("utf8"))
            ## Remove stop words
            stops = set(nltk.corpus.stopwords.words("english"))
            text = [w for w in text if not w in stops]
            
            text = " ".join(text)
            ## Stemming
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

            
            
            #find longest tweet
            if len(text)> MAX_NB_WORDS:
                MAX_NB_WORDS = len(text)
            #translator function does not work with utf8
            texts_train.append(" ".join(text).encode('ascii'))
            labels_train.append(row[2])

        print('Found %s train texts.' % len(texts_train))
        #read dev data
        texts_dev = []  # list of text samples

        labels_dev = []  # list of label ids
        for row in self.corpus_data_dev:
            ## Convert words to lower case and split them
            text = row[3].lower()
            text = tweet_tokenizer.tokenize(unicode(row[3]).encode("utf8"))
            ## Remove stop words
            stops = set(nltk.corpus.stopwords.words("english"))
            text = [w for w in text if not w in stops]
            
            text = " ".join(text)
            ## Stemming
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)
            #find longest tweet
            if len(text)> MAX_NB_WORDS:
                MAX_NB_WORDS = len(text)
            texts_dev.append(" ".join(text).encode('ascii'))
            labels_dev.append(row[2])

        print ('Found %s dev texts.' % len(texts_dev))
        

        #transform texts into tensors
        tokenizer = Tokenizer(num_words=400000,filters='')
        tokenizer.fit_on_texts(texts_train+texts_dev)
        sequences_train = tokenizer.texts_to_sequences(texts_train)
        sequences_dev = tokenizer.texts_to_sequences(texts_dev)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        x_train = pad_sequences(sequences_train, maxlen=MAX_NB_WORDS)
        x_val =   pad_sequences(sequences_dev, maxlen=MAX_NB_WORDS)
        
        y_train = to_categorical(np.asarray(labels_train))
        y_val= to_categorical(np.asarray(labels_dev))
        print('Shape of train data tensor:', x_train.shape)
        print('Shape of train label tensor:', y_train.shape)

        #use pretrained embeddings

        embeddings_index = self._load_embedding_vectors_glove()
#        vocabulary_size=400000
#        embedding_matrix = np.zeros((vocabulary_size, 100))
#        for word, index in tokenizer.word_index.items():
#                if index > vocabulary_size - 1:
#                    break
#                else:
#                    embedding_vector = embeddings_index.get(word)
#                    if embedding_vector is not None:
#                        embedding_matrix[index] = embedding_vector
#        for line in self.embeddings:
#            values = line.split()
#            word = values[0]
#            coefs = np.asarray(values[1:], dtype='float32')
#            embeddings_index[word] = coefs
       

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, 100))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                
        

        

        return x_train,y_train,x_val,y_val,embedding_matrix,MAX_NB_WORDS,word_index

    def _load_embedding_vectors_glove(self):
        embeddings_index = dict()
        
        for line in self.embeddings:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        self.embeddings.close()
        return embeddings_index


                    
class SimpleReader:
    def __init__(self, path_to_file):
        tsvfile = codecs.open(path_to_file)
        self.corpus_data = csv.reader(tsvfile, delimiter='\t')
        self.corpus = []
    
    def prepare_corpus(self):
                      
        #################
        # PREPROCESSING #
        #################
        for row in self.corpus_data:
        
            # add the tokenized text using a dedicated Tweet tokenizer
            tweet_tokenizer = TweetTokenizer()
            tweet= tweet_tokenizer.tokenize(row[3])
            lemmatized=''
            lmt = WordNetLemmatizer()
            for word in tweet:
                lemmatized+=lmt.lemmatize(word)+' '
           
            self.corpus.append((lemmatized.strip(),row[2]))
       
        return self.corpus
 

class SubjectivityClueReader:
    
     
    # this is the constructor of the SubjectivityClueReader class. It makes the
    # subjectivity clues available to the class
    def __init__(self, path_to_file):
        tsvfile = codecs.open(path_to_file)
        self.subjclue_data = csv.reader(tsvfile, delimiter=' ')
        self.subjclues_neg = {}
        self.subjclues_pos = {}
        self.subjclues_neu = {}
        self.read_lexicon()
  
    def read_lexicon(self):
    
        for row in self.subjclue_data:
            try:
                if row[5].split("=")[1] =="negative":
                    self.subjclues_neg[row[2].split("=")[1]] ={"pos":row[3].split("=")[1],"type":row[0].split("=")[1]}
                elif row[5].split("=")[1] =="positive":
                    self.subjclues_pos[row[2].split("=")[1]] ={"pos":row[3].split("=")[1],"type":row[0].split("=")[1]}
                elif row[5].split("=")[1] =="neutral":
                    self.subjclues_neu[row[2].split("=")[1]] ={"pos":row[3].split("=")[1],"type":row[0].split("=")[1]}
            except:
                continue
                    
        
class OpinionLexiconReader:

    # this is the constructor of the OpinionLexiconReader class. It makes the
    # opinion lexicon file available to the class
    def __init__(self, path_to_file):
        self.lexfile = codecs.open(path_to_file,'r','utf8').readlines()
        self.opinion_words=[]
        self.read_lexicon()
  
    def read_lexicon(self):
    
        for line in self.lexfile:
           
            if line.strip() !="" and line[0] !=';':
                self.opinion_words.append(line.strip())
                    
                    
                    
