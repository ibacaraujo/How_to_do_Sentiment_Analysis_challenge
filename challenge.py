
# coding: utf-8

# In[1]:

# load the dependencies
import numpy as np
import pandas as pd
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split


# In[2]:

# load the data set
ign = pd.read_csv('ign.csv', index_col=0)
ign.head()


# In[3]:

# binarize the classes
ign.loc[ign['score_phrase'] == 'Amazing', 'score_phrase'] = 1
ign.loc[ign['score_phrase'] == 'Great', 'score_phrase'] = 1
ign.loc[ign['score_phrase'] == 'Good', 'score_phrase'] = 1
ign.loc[ign['score_phrase'] == 'Masterpiece', 'score_phrase'] = 1
ign.loc[ign['score_phrase'] == 'Okay', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Mediocre', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Painful', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Awful', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Bad', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Unbearable', 'score_phrase'] = 0
ign.loc[ign['score_phrase'] == 'Disaster', 'score_phrase'] = 0
ign.head()
ign['score_phrase'].value_counts()


# In[4]:

# Split the data into train and test set
X = ign.drop(['score_phrase', 'title', 'url', 'platform', 'genre', 'editors_choice', 'release_year', 'release_month', 'release_day'], axis=1, inplace=False)
X = np.array(X)
Y = ign['score_phrase'].to_frame()
Y = np.array(Y)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.33)


# In[5]:

# Data preprocessing
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


# In[6]:

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=12478, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')


# In[ ]:

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)

