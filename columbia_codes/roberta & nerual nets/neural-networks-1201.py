#!/usr/bin/env python
# coding: utf-8

# # End model to predict financial incentives and disincentives
# 
# John Brandt
# 
# Last updated: Aug 19, 2019
# 
# 
# This notebook contains a gold standard baseline (LSTM with gold standard labels) as well as a noisy implementation of snorkel labels with roBERTa encoding.

"""
This file includes:
1. a modified Dataset and Model function which can take in additional features
2. modified method to calculate evaluation scores on validation dataset
3. additional method to calculate evaluation scores on test dataset
4. an early-stopping method
"""

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

# ## Shared classes

# #### Data Loader

# In[2]:


from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels, mode, add_features):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = mode
        self.add_features = add_features

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # concatenate additional features
        X = np.load('/home/yg2619/capstone/data/' + self.mode + '_embeddings/' + str(ID) + '.npy')
        add = np.array(self.add_features.loc[ID])
        y = self.labels[ID]

        return X.reshape((50, 1024)), y, add


# In[3]:


# load updated gold standard datasets and snorkel labels dataset
gs = pd.read_csv("~/columbia_codes/data/gold_standard_updated.csv")
gs_probas = np.load('~/columbia_codes/data/snorkel_proba_updated.npy')
noisy_probas = np.load('~/columbia_codes/data/snorkel_noisy_proba_updated.npy')
# load embeddings dataset
gs_embeddings = np.load('~/columbia_codes/data/test_embedding.npy')
noisy_embeddings = np.load('~/columbia_codes/data/train_embedding.npy')
# load additional features
gs_features = pd.read_csv("~/columbia_codes/data/gs_allFeatures.csv")
noisy_features = pd.read_csv("~/columbia_codes/data/noisy_allFeatures.csv")

# reduced the original batch size from 50 to 20
params = {'batch_size': 20,
          'shuffle': True,
          'num_workers': 2}

# Y = df['class'] - 1
#
# # Datasets
# partition = {'train': [x for x in range(0, 800)],
#              'validation': [x for x in range(800, 900)],
#              'test': [x for x in range(900, 1000)]}
#
# noisy_labels = {k: w for w, k in zip(noisy_probas, range(1000))}
# gs_labels = {k: w for w, k in zip(Y, range(800, 1000))}
#
# # Generators
# training_set = Dataset(partition['train'], gs_probas[0:800], 'test', gs_features)
# training_generator = data.DataLoader(training_set, **params)
#
# validation_set = Dataset(partition['validation'], gs_labels, 'test', gs_features)
# validation_generator = data.DataLoader(validation_set, **params)
#
# test_set = Dataset(partition['test'], gs_labels, 'test', gs_features)
# test_generator = data.DataLoader(test_set, **params)


# #### Soft label loss

# In[4]:


class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float) targets
    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses
    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        if weight is None:
            self.weight = None
        else:
            # Note: Sets the attribute self.weight as well
            self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

    def forward(self, input, target):
        n, k = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y].float() * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


# #### Training function

# In[5]:


def apply(model, criterion, batch, targets, feature):
    pred = model(torch.autograd.Variable(batch), feature)
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train(model,
          n_epochs,
          train_generator,
          val_generator):
    softcriterion = SoftCrossEntropyLoss()
    hardcriterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    counter = 0
    valid_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = torch.zeros(1)
        valid_loss = torch.zeros(1)
        for inputs, labels, features in train_generator:
            model.train()
            inputs = inputs.type(torch.FloatTensor)
            features = features.type(torch.FloatTensor)
            pred, loss = apply(model, softcriterion, inputs, labels, features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data  # changed the train loss into a total loss

        # Get validation loss
        with torch.no_grad():
            val_h = model.init_hidden(batch_size)
            model.eval()
            for inputs, labels, features in val_generator:
                inputs = inputs.type(torch.FloatTensor)
                features = features.type(torch.FloatTensor)
                pred, loss = apply(model, softcriterion, inputs, labels, features)
                valid_loss += loss.data  # changed the validation loss into a total loss

        print("Epoch: {}/{}...".format(epoch, n_epochs),
              "Step: {}...".format(counter),
              "Loss: {:.6f}...".format(train_loss.item()),
              "Val Loss: {:.6f}".format(valid_loss.item()))

        # early stopping method
        if len(valid_losses) < 1 or valid_loss.item() <= valid_losses[-1]:  # if the dev loss continues to decline
            counter = 0
            valid_losses.append(valid_loss.item())  # add the current loss to the list
        elif valid_loss.item() > valid_losses[-1] and counter < 3:
            counter += 1
        else:
            break  # otherwise, if the dev loss is not improving, stop the iteration

    return model


# add a test method to calculate test loss and f1 score
def test(model, loss_fn, test_generator):
    gold = []
    predicted = []

    loss = torch.zeros(1)
    model.eval()

    with torch.no_grad():
        for X_b, y_b, features in test_generator:
            X_b = X_b.type(torch.FloatTensor)
            features = features.type(torch.FloatTensor)
            y_pred = model(X_b, features)
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.reshape((batch_size, 2)).argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.reshape((batch_size, 2)), y_b.long()).data

    print("Test loss: ")
    print(loss.item())
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))
    print("Accuracy Score: ")
    print(accuracy_score(gold, predicted))
    print('ROC AUC Score: ')
    print(roc_auc_score(gold, predicted, average='macro'))
    # Matthews Correlation Coefficient is a good measure for evaluating unbalanced datasets
    print('Matthews Correlation Coefficient: ')
    print(matthews_corrcoef(gold, predicted))


# ## Gold standard baseline
# 
# Shallow RNN with RoBERTa encoded words.

# In[6]:


import torch.autograd as autograd


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, feature_size):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim+feature_size, self.output_size)

    def forward(self, batch, feature):
        # changed -2 to -3, should be equal to the batch_size
        self.hidden = self.init_hidden(batch.size(-3))
        outputs, (ht, ct) = self.lstm(batch, self.hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = torch.cat((ht[-1], feature), 1)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)))


# In[7]:


# model = Model(input_size=1024, output_size=2, hidden_dim=200, n_layers=1)
# model.to("cpu")
#
# n_epochs = 100
#
# # In[8]:
#
#
# device = "cpu"
# print_every = 100
# batch_size = 50
# loss_fn = nn.CrossEntropyLoss()
#
# model = train(model=model,
#               n_epochs=250,
#               train_generator=training_generator,
#               val_generator=validation_generator)
# # model = torch.load('/home/yg2619/capstone/baseline_new.pth')
# # save the model for later use
# torch.save(model, 'baseline_new.pth')
# # test the model performance
# test(model, loss_fn, test_generator)

# Snorkel end model with RoBERTA-LSTM input module and noise-aware output head

# In[11]:


Y = gs['class'] - 1

# Datasets
partition = {'train': [x for x in range(0, 12000)],
             'validation': [x for x in range(12000, 15000)],
             'test': [x for x in range(0, 1000)]}

noisy_labels = {k: w for w, k in zip(noisy_probas, range(15000))}
gs_labels = {k: w for w, k in zip(Y, range(0, 1000))}

# Generators
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

training_set = Dataset(partition['train'], noisy_labels, 'train', noisy_features)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], noisy_labels, 'train', noisy_features)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(partition['test'], gs_labels, 'test', gs_features)
test_generator = data.DataLoader(test_set, **params)

# In[12]:


model = Model(input_size=1024, output_size=2, hidden_dim=200, n_layers=2, feature_size=noisy_features.shape[1])
model.to("cpu")

loss_fn = nn.CrossEntropyLoss()

n_epochs = 100

device = "cpu"
print_every = 100
batch_size = 20

model = train(model=model,
              n_epochs=5,
              train_generator=training_generator,
              val_generator=validation_generator)
torch.save(model, 'sixth_model.pth')
# model = torch.load('/home/yg2619/capstone/fifth_model.pth')
test(model, loss_fn, test_generator)

# ## To do: Snorkel end model with RoBERTa-LSTM input module and multi-task output head

# ## To do: Snorkel end model with RoBERTa-LSTM input module and multi-task output head, with concatenation of additional feature engineering

# ## To do: Snorkel end model with RoBERTa-LSTM input module and multi-task output head, with concatenation of additional feature engineering and synonym augmentation
