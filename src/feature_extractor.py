# this is where the features are extracted
# -*- coding: utf-8 -*-

import re
from data_reader import OpinionLexiconReader,SubjectivityClueReader

class FeatureExtractor:

    # this is the constructor of the FeatureExtractor class
    def __init__(self):
        print "Starting feature extraction"
        
    # THIS IS WHERE THE DIFFERENT FEATURE EXTRACTION FUNCTIONS ARE CALLED #

    def extract_features(self,corpus,pos_opinion,neg_opinion,subjectivity_clues):
        # featureset is a list
        featureset = list()
        
        #load the external lexicons into memory
        ol_pos=OpinionLexiconReader(pos_opinion)
        ol_neg=OpinionLexiconReader(neg_opinion)
        opinion_words= ol_pos.opinion_words+ol_neg.opinion_words
        subjectivity_lexs=SubjectivityClueReader(subjectivity_clues)
        
        # this for-loop loops through every tweet in the corpus    
        #and appends the features and the annotation/label of tweet the featureset list (as seen above in the example)
        for tweet_id in corpus:
            #initally the featureset is empty
            tweet = corpus[tweet_id]
            featureset.append(({},tweet['label']))
                    

            ####################################################################################
            # THIS IS WHERE ALL THE DIFFERENT POSSIBLE FEATURE EXTRACTION FUNCTIONS ARE CALLED #
            ####################################################################################
            # structure of feature function for example of the feature "number_of_words":
            # -1 calls the last tweet that has been appended to the featureset
            # 0 accesses the feature dictionary which is the first element of the tupel
            # "number of words" is the feature name

            featureset[-1][0]["number_of_words"] = self.number_of_words(tweet)
            featureset[-1][0]["number_of_pos_opinion"] = self.number_of_pos_opinion(tweet,ol_pos)
            featureset[-1][0]["number_of_neg_opinion"] = self.number_of_pos_opinion(tweet,ol_neg)
            for word in opinion_words:
                if word in tweet["lemmatized_tweet"]:
                    featureset[-1][0][word] = 1
                else:
                    featureset[-1][0][word] = 0
            featureset[-1][0]["number_of_adjectives"] = self.number_of_adjectives(tweet)
            featureset[-1][0]["number_of_ly_adverbs"] = self.number_of_ly_adverbs(tweet)
            featureset[-1][0]["number_of_exclamation_marks"] = self.number_of_exclamation_marks(tweet)
            featureset[-1][0]["number_of_weaksubj_neg"] = self.number_of_weaksubj_neg(tweet,subjectivity_lexs.subjclues_neg)
            featureset[-1][0]["number_of_weaksubj_pos"] = self.number_of_weaksubj_pos(tweet,subjectivity_lexs.subjclues_pos)
            featureset[-1][0]["number_of_weaksubj_neu"] = self.number_of_weaksubj_neu(tweet,subjectivity_lexs.subjclues_neu)
            featureset[-1][0]["number_of_strongsubj_neg"] = self.number_of_strongsubj_neg(tweet,subjectivity_lexs.subjclues_neg)
            featureset[-1][0]["number_of_strongsubj_pos"] = self.number_of_strongsubj_pos(tweet,subjectivity_lexs.subjclues_pos)
            featureset[-1][0]["number_of_strongsubj_neu"] = self.number_of_strongsubj_neu(tweet,subjectivity_lexs.subjclues_neu)
            featureset[-1][0]["number_of_names"] = self.number_of_names(tweet)
                    
        return featureset


    #####################################################################
    # THESE ARE ALL THE DIFFERENT POSSIBLE FEATURE EXTRACTION FUNCTIONS #
    #####################################################################

    # This function returns the number of words in the tweet
    def number_of_words(self, word_dic):
        
        return len(word_dic["lemmatized_tweet"])


    # This function returns the number of positive opinion words in the tweet
    def number_of_pos_opinion(self, word_dic,ol):
        
        count=0
        for word in word_dic["lemmatized_tweet"]:
            if word in ol.opinion_words:
                count+=1
   
        return count
        
    # This function returns the number of negative opinion words in the tweet
    def number_of_pos_opinion(self, word_dic,ol):
        
        count=0
        for word in word_dic["lemmatized_tweet"]:
            if word in ol.opinion_words:
                count+=1
   
        return count
               
    # This function returns the number of adjectives (which often hold sentiment) in the tweet
    def number_of_adjectives(self, word_dic):
        
        poslist = [pos[1] for pos in word_dic["pos"]]

        return poslist.count("JJ")
        
   # This function returns the number of -ly adverbs (totally, completely...) in the tweet
    def number_of_ly_adverbs(self, word_dic):
        
        count=0
        for word in word_dic["pos"]:
            if word[0][-2:]=="ly" and word[1]=="RB":
                count+=1
   
        return count
   
    # This function returns the number of exclamation marks in the tweet
    def number_of_exclamation_marks(self, word_dic):
        
        count=0
        for word in word_dic["lemmatized_tweet"]:
            if word =="!":
                count+=1
   
        return count
    
    # This function returns the number of negative subjectivity clues with weak subjectivity
    def number_of_weaksubj_neg(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="weaksubj":
                count+=1
        
        return count


    # This function returns the number of positive subjectivity clues with weak subjectivity
    def number_of_weaksubj_pos(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="weaksubj":
                count+=1
        
        return count


    # This function returns the number of neutral subjectivity clues with weak subjectivity
    def number_of_weaksubj_neu(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="weaksubj":
                count+=1
        
        return count
                
    # This function returns the number of negative subjectivity clues with strong subjectivity
    def number_of_strongsubj_neg(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="strongsubj":
                count+=1
        
        return count

    # This function returns the number of positive subjectivity clues with strong subjectivity
    def number_of_strongsubj_pos(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="strongsubj":
                count+=1
        
        return count
        
    # This function returns the number of neutral subjectivity clues with strong subjectivity
    def number_of_strongsubj_neu(self,word_dic,subjclues):
       
        count = 0
        for word in word_dic["lemmatized_tweet"]:
            if word in subjclues and subjclues[word]["type"]=="strongsubj":
                count+=1
        
        return count
        
    # This function returns the number of names found in the tweet, the assumption is that 
    # people express sentiment with respect to people, thus --> not neutral
    def number_of_names(self,word_dic):
       
        count = 0
        chunk=str(word_dic["chunked"])
        personlist =re.findall("PERSON",chunk)
        
        return len(personlist)
