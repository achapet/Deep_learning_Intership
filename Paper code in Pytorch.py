
# coding: utf-8

#Import needed libraries
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import random
import os
import pickle
import theano

#Importing Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

#Data plotting
from pandas.plotting import radviz
from pandas.plotting import parallel_coordinates

import pytoune
from pytoune.framework import Model
from pytoune.framework import callbacks


# # Import data

# Buid the feature matrix
data = pd.read_csv('/Users/almachapet--batlle/Documents/Internship U1001/2017---Deep-learning-yeast-UTRs-master/Data/Random_UTRs.csv')
print(data)


# ## One-hot encoding of the sequences.
# 
# i.e. we're converting the sequences from being represented as a 50 character string of bases to a 4x50 matrix of 1's and 0's, with each row corresponding to a base and every column a position in the UTR.


# From the work of Cuperus et al.
# one hot encoding of UTRs
# X = one hot encoding matrix
# Y = growth rates

def one_hot_encoding(df, seq_column, expression):

    bases = ['A','C','G','T']
    base_dict = dict(zip(bases,range(4))) # {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

    n = len(df)
    
    # length of the UTR sequence
    # we also add 10 empty spaces to either side
    total_width = df[seq_column].str.len().max() + 20
    
    # initialize an empty numpy ndarray of the appropriate size
    X = np.zeros((n, 1, 4, total_width))
    
    # an array with the sequences that we will one-hot encode
    seqs = df[seq_column].values
    
    # loop through the array of sequences to create an array that keras will actually read
    for i in range(n):
        seq = seqs[i]
        
        # loop through each individual sequence, from the 5' to 3' end
        for b in range(len(seq)):
            # this will assign a 1 to the appropriate base and position for this UTR sequence
            X[i, 0, base_dict[seq[b]], int(b + round((total_width - len(seq))/2.))] = 1.
    
        # keep track of where we are
        if (i%10000)==0:
            print(i),
        
    X = X.astype(theano.config.floatX)
    Y = np.asarray(df[expression].values,
                   dtype = theano.config.floatX)[:, np.newaxis]
    
    return X, Y, total_width


X, Y, total_width = one_hot_encoding(data, 'UTR', 'growth_rate')

print(X)
print(Y)


# ## Generate different data sets

# a sorted numpy array of UTR indexes, from least reads to most reads
sorted_inds = data.sort_values('t0').index.values


train_inds = sorted_inds[:int(0.95*len(sorted_inds))] # 95% of the data as the training set


test_inds = sorted_inds[int(0.95*len(sorted_inds)):] # UTRs with most reads at time point 0 as the test set

# set the seed before randomly shuffling the data
seed = 0.5
random.shuffle(train_inds, lambda :seed)


# # Generate Model
# 
# I need to figure out how to make the dropout happen and Flatten. 
# How do hidden units work in fully connected layers?

# ## Buid the neural network
# 
# Try different structures

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input channel, output channels = number of filters, convolution kernel size
        # kernel
        self.conv1 = nn.Conv2d(400, 128, [4,13])
        self.conv2 = nn.Conv2d(400, 16, [1,13])
        self.conv3 = nn.Conv2d(400, 16, [1,13])
        self.fc1 = nn.Linear(400, 12)
        self.lin_out1 = nn.Linear(120, 400)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        x = self.lol1(x)
        x = nn.Dropout(p=0.15) #

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


# In[70]:


params = list(net.parameters())
print(len(params))
print(params[0].size()) 


# ## Training

# In[77]:


# Choice of optimizer & loss function => MSE 
# Using backpropagation

# define model
model = Net()

# define loss function
loss_func = nn.MSELoss() 

# define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

#Verification
for epoch in range(2):  # loop over the dataset multiple time
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')


# ## Training with PyToune

# In[93]:


num_features = 20
num_epochs = 10
num_train_samples = 800
batch_size = 20

loss_function = torch.nn.MSELoss()
pytorch_module = torch.nn.Linear(num_features,1)
optimizer = torch.optim.Adam(pytorch_module.parameters(), lr=1e-3)
    
Model = Net()
    
# track model overfitting
earlyStopping = pytoune.framework.EarlyStopping(monitor = 'val_loss',
                                                  patience = 1,
                                                  verbose = 0,
                                                  mode = 'min')
    
# fit the model
# note that I'm not passing the data to this function, I've just included it here (i.e. I've
# included X and Y)
Model.train(X[train_inds],
              #Y[train_inds],
              #callbacks = [earlyStopping],
              #verbose = 0,
              #nb_epoch = 100,
           )
    
#print ('MSE:',earlyStopping.best)
return {'loss': earlyStopping.best, 'status': STATUS_OK}


# ## Plot predictions vs data

# In[ ]:


Y_pred = model.predict(X, verbose=1)


# In[ ]:


# data
x = Y_pred[test_inds].flatten()
y = Y[test_inds].flatten()

# calculate R^2
r2 = scipy.stats.pearsonr(x, y)[0]**2


g = sns.jointplot(x,
                  y,
                  stat_func = None,
                  kind = 'scatter',
                  s = 5,
                  alpha = 0.1,
                  size = 5)

g.ax_joint.set_xlabel('Predicted log$_2$ Growth Rate')
g.ax_joint.set_ylabel('Measured log$_2$ Growth Rate')


text = "R$^2$ = {:0.2}".format(r2)
plt.annotate(text, xy=(-5.5, 0.95), xycoords='axes fraction')

plt.title("CNN predictions vs. test set", x = -3, y = 1.25)

