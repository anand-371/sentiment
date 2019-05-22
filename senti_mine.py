import pandas as pd
data = pd.read_csv("train.csv",encoding='ISO-8859-1')
import  numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
import pickle
def create_vectorizer(pos_txt,neg_txt):
    '''
    creating the vectorizer object with all the words from the dataset
    '''
    lexicon = []   #all words in the list
    lexicon.extend(pos_txt)
    lexicon.extend(neg_txt)
    vectorizer =CountVectorizer(max_features=1000)  #vectorizer object
    vectorizer.fit(lexicon)
    return vectorizer

def sample_handling(sample,vectorizer,classification):
    featureset=[] #list of features and its classification
    vectweet = vectorizer.transform(sample).toarray()
    for tweet in vectweet:
        featureset.append([list(tweet),classification])
    print('almost done')
    return featureset
def create_sample_sets_and_labels(test_size=0.1):
            pos_txt=list(data.SentimentText[data.Sentiment==1])[:5000]
            neg_txt=list(data.SentimentText[data.Sentiment==0])[:5000]
            vectorizer=create_vectorizer(pos_txt,neg_txt)
            features=[]
            features+=sample_handling(pos_txt,vectorizer,[1,0])
            features+=sample_handling(neg_txt,vectorizer,[0,1])
            random.shuffle(features)
            features=np.array(features)
            testing_size=int(test_size*len(features))
            train_x = list(features[:,0][:-testing_size])
            train_y = list(features[:,1][:-testing_size])
            test_x = list(features[:,0][-testing_size:])
            test_y = list(features[:,1][-testing_size:])
            return train_x,train_y,test_x,test_y


train_x,train_y,test_x,test_y= create_sample_sets_and_labels()

