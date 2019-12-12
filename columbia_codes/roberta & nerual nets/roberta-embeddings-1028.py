#!/usr/bin/env python
# coding: utf-8

# # roBERTa paragraph encodings
# 
# Encodes paragraphs with roberta embedding layer for input into snorkel end model.
# 
# John Brandt
# 
# 
# Last Updated: August 21, 2019

# In[2]:


import torch
import pandas as pd

# # TODO: Fine tune roBERTa

# In[1]:


roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

# ## Encode sentences with roberta

# In[29]:


df = pd.read_csv("/data/noisy.csv")
gs = pd.read_csv("/data/gold_standard.csv")

# In[36]:


## This cell needs to be run for both df and gs

from tqdm import tnrange
import time
import numpy as np


def encoder(sentence, max_len=50):
    sentence_split = sentence.split(' ')
    sentence = ' '.join(sentence_split[:(min(len(sentence_split), max_len))])
    tokens = roberta.encode(sentence)
    tokens = tokens[:min(len(tokens), max_len)]
    # Extract all layer's features (layer 0 is the embedding layer)
    embedding = roberta.extract_features(tokens, return_all_hiddens=True)[0]
    if embedding.shape[1] < max_len:
        padding = torch.zeros(1, max_len - embedding.shape[1], 1024, dtype=torch.float)
        embedding = torch.cat((embedding, padding), 1)
    return embedding


for i in tnrange(0, len(df)):
     encoding = encoder(df['sentences'][i])
     encoding = encoding.detach().numpy()
     np.save("/data/train_embeddings/" + str(i), encoding)

for i in tnrange(0, len(gs)):
     encoding = encoder(gs['sentences'][i])
     encoding = encoding.detach().numpy()
     np.save("/data/test_embeddings/" + str(i), encoding)


# In[14]:


# Defunct

encodings = []
for i in tnrange(0, len(df)):
 encodings.append(np.load("/data/train_embeddings/" + str(i) + ".npy"))

encodings_stacked = np.concatenate(encodings)
np.save('/data/encodings', encodings_stacked)
