from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from nltk import word_tokenize, pos_tag
from collections import Counter
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd

# load dataset
noisy = pd.read_csv("~/capstone/data/noisy.csv")
gs = pd.read_csv("~/capstone/data/gold_standard.csv")
noisy_probas = np.load('/home/yg2619/capstone/data/snorkel_noisy_proba_updated.npy')
noisy_labels = noisy_probas.argmax(1)

# use pipeline to generate n-gram
pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('scaler', StandardScaler(with_mean=False)),
    ('kbest', SelectKBest(k=200))
])

gs_ngram = pipeline.fit_transform(gs['sentences'], gs['class'])
noisy_ngram = pipeline.fit_transform(noisy['sentences'], noisy_labels)


# generate pos tags
# by looking into the most common pos tags, we decided to use the count of pos tags
# 'NN', 'IN', 'JJ', 'NNS', 'DT', 'CC', 'CD', 'VB' as the new pos tag feature for each sentence
def generate_pos(text):
    pos_col = ['countsofNN', 'countsofIN', 'countsofJJ', 'countsofNNS', 'countsofDT', 'countsofCC', 'countsofCD',
               'countsofVB']
    pos_tags = pd.DataFrame(columns=pos_col)
    for i in range(len(text)):
        sen = text[i]
        tokens = word_tokenize(sen)
        tags = pos_tag(tokens)
        counts = Counter([tag for word, tag in tags])
        for col in pos_col:
            pos = col.strip('countsof')
            pos_tags.loc[i, col] = counts[pos]
    # scale the pos tag counts
    scaler = StandardScaler()
    pos_tags = pd.DataFrame(scaler.fit_transform(pos_tags))
    # rename the columns
    pos_tags.columns = pos_col
    return pos_tags


gs_pos = generate_pos(gs['sentences'])
noisy_pos = generate_pos(noisy['sentences'])

# combine pos tags, n-gram, topic modeling, and sentiment score features
gs_features = pd.read_csv('~/capstone/data/gs_features.csv')
noisy_features = pd.read_csv('~/capstone/data/noisy_features.csv')

gs_ngram = pd.DataFrame(gs_ngram.toarray())
gs_new_features = pd.concat([gs_pos, gs_features['sentiscore'], gs_ngram], axis=1, sort=False)
gs_new_features.to_csv('~/capstone/data/gs_newFeatures.csv', index=False)

noisy_ngram = pd.DataFrame(noisy_ngram.toarray())
noisy_new_features = pd.concat([noisy_pos, noisy_features['sentiscore'], noisy_ngram], axis=1, sort=False)
noisy_new_features.to_csv('~/capstone/data/noisy_newFeatures.csv', index=False)
