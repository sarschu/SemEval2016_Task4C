# SemEval-2016 Task 4 C

This repository contains python code to train different systems for the 
polarity classification on a 5-point scale of Tweets and an evaluation and 
comparison of the suggested approaches.

The task has been suggested in 2016's [SemEval Shared Task 4C](alt.qcri.org/semeval2016/task4/ "SemEval Shared Task"). 
All systems are trained using the training data provided by the organizers. 

## Data 

The original dataset contains 10,000 Twitter posts coming from 100 different topics (100 posts per topic). The dataset was split in  

train 6,000  
dev   2,000  
devtest 2,000  

The data is provided in form of a file containing the Twitter status IDs. The actual posts can be downloaded 
using these provided files in combination with a [download script](https://github.com/aritter/twitter_download "Tweet downloader").  

Two years later, not all tweets are available for download anymore which results in the following numbers for the dataset:  
train 2,245  
dev   696  
devtest 1,081  

This also makes a direct comparison of the results achieved here and in the original shared task impossible.  

The tweets are annotated with a polarity value ranging from -2 to 2, where -2 corresponds to very negative, 
-1 corresponds to negative, 0 corresponds to neutral, 1 corresponds to positive and 2 corresponds to very positive.  

This results in a file with one annotated tweet per line of the following format:  

``
631104156187627520	@microsoft	-1	For the 1st time @Skype has a "High Startup impact" Does anyone at @Microsoft have a clue? #Windows10Failpic.twitter.com/loO3yd5rwe
``

In order to get an impression on the composition of the dataset, I had a look at the label distribution in the training data (using data_statistics.py):  

Number of very negative Tweets  
19  
Number of  negative Tweets  
204  
Number of neutral Tweets  
555  
Number of positive Tweets  
1248  
Number of very positive Tweets  
171  

It turns out that the data is heavily skewed towards the positive class (similar picture for dev set) which has to be kept in mind.  

## System requirements
python2.7  
nltk  
nltk.corpus.stopwords  
nltk.tokenize  
numpy  
nltk.stem.wordnet   
nltk.stem.snowball  
keras  
theano  
configparser  
sklearn  

## Suggested systems

We compare the results of three different system:

 * Naive Bayes classifier only based on bag of word features 
 * Support Vector Machine classifier with linguistically informed features
 * Recurrent neural net with pre-trained word embeddings (Glove)

### Preprocessing

Before training the tweets are preprocessed in the following way:
    1) tokenized with nltk TweetTokenizer (accounts for smileys, hashtags and other Tweet specific phenomena)
    2) lemmatized with nltk's WordNetLemmatizer

### Features

For the training of the Naive Bayes classifier and the Support Vector Machines different feature sets are used.  

The **Naive Bayes** classifier (nltk) relies on the presence and/or absence of lexical items (lemmas). The conditional probabilities of words with respect to a polarity label are learned from the training data.  

For the training of the **Support Vector Machine** classifier (nltk wrapper for sklearn), in addition to the lexical features, linguistically more complex features as well as features extracted from sentiment and subjectivity clue lexicons are extracted. The implemented features are described in ``feature_extractor.py``

I used external lexicons to enrich the features extracted for each tweet

* Opinion lexicon [source](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)
* Subjectivity Clues [source](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)

The **Recurrent Neural Net** (keras) uses pre-trained word embeddings [Glove](https://nlp.stanford.edu/projects/glove/ "Glove for Twitter") for Twitter with 100 dimensions. The network architecture consists of the embedding layer, a 1d convolutional layer, a max pooling layer and an LSTM layer (100), the output layer is a dense layer with sigmoid activation.  I trained it for 3 epochs using categorical crossentropy and adam.


## Evaluation

As evaluation metrics we use F1 Score (following the Shared Task)

| Classifier             | Development | Test |  
|------------------------|-------------|------|  
| Baseline               | 0.509       | 0.517|  
| Naive Bayes            | 0.53        | 0.515|  
| Support Vector Machine | 0.509       | 0.517|  
| Recurrent Neural Net   | 0.526       | 0.535|  

