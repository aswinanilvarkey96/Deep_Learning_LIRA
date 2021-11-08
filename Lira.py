# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:56 2021

@author: aswin
"""

# import pandas as pd

# object = pd.read_pickle(r'aligned_pass-0_cleaned.pickle')

import pickle

with open('aligned_pass-0_cleaned.pickle', 'rb') as f:
    data = pickle.load(f)
    
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

columns = data.columns
z_data = data['GM.acc.xyz.z']
arr = data['GM.acc.xyz.z'][0]
acc_z_cat = []
acc_x_cat = []
acc_y_cat = []
vel_cat = []
s = []
vel_itr = []
max = np.max(data['IRI_mean'][0])
idx = 0
l_x = []
l_y = []
l_z = []
v_len = []
for i in range(len(data['GM.acc.xyz.z'])):
    # acc_z = np.sum(data['GM.acc.xyz.z'][i])/len(data['GM.acc.xyz.z'][i])
    # acc_x = np.sum(data['GM.acc.xyz.x'][i])/len(data['GM.acc.xyz.x'][i])
    # acc_y = np.sum(data['GM.acc.xyz.y'][i])/len(data['GM.acc.xyz.y'][i])
    # vel = np.sum(data['GM.obd.spd_veh.value'][i])/len(data['GM.obd.spd_veh.value'][i])
    acc_z = np.max(data['GM.acc.xyz.z'][i])
    acc_x = np.max(data['GM.acc.xyz.x'][i])
    acc_y = np.max(data['GM.acc.xyz.y'][i])
    vel = np.max(data['GM.obd.spd_veh.value'][i])
    acc_z_cat.append(acc_z)
    acc_x_cat.append(acc_x)
    acc_y_cat.append(acc_y)
    vel_cat.append(vel)
    v = [np.ones(len(data['GM.acc.xyz.y'][i]))*vel]
    vel_itr.append(v)
    if(np.max(data['IRI_mean'][i])> max):
        max = np.max(data['IRI_mean'][i])
        idx = i
    s.append(len(data['GM.acc.xyz.y'][i]))
    l_x.append(len(data['GM.acc.xyz.x'][i]))
    l_y.append(len(data['GM.acc.xyz.y'][i]))
    l_z.append(len(data['GM.acc.xyz.z'][i]))
    v_len.append(len(data['GM.obd.spd_veh.value'][i]))
    
    
data_avg = {'z': acc_z_cat,
        'x': acc_x_cat,
        'y': acc_y_cat,
        'vel':vel_cat
        }

df = pd.DataFrame(data_avg,columns=['z','x','y','vel'])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

plt.figure()
plt.hist(vel_cat)

plt.figure()
plt.plot(l_x)
plt.plot(l_y)
plt.plot(l_z)
plt.legend(['x','y','z'])
#%%
plt.figure()
plt.scatter(vel_cat,acc_z_cat,s=9,alpha=0.5)

plt.figure()
plt.plot((data['GM.acc.xyz.z'][idx]))
plt.figure()

#%%

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
 
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=60
        )
        self.encoder_output_layer = nn.Linear(
            in_features=60, out_features=60
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=60, out_features=60
        )
        self.decoder_output_layer = nn.Linear(
            in_features=60, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        return reconstructed
    
LEARNING_RATE = 3e-4
criterion = nn.MSELoss()   #<-- Your code here.   

BATCH_SIZE = 100
net = AE(input_shape=BATCH_SIZE)
# weight_decay is equal to L2 regularization
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

train_dataset = moving_average(data['GM.acc.xyz.y'][8],10)

test_dataset = moving_average(data['GM.acc.xyz.y'][1615],10)

num_batches_train = len(train_dataset)/BATCH_SIZE

#%%
epochs = 30
plt.figure()
iri = np.asarray([])
plt.plot(train_dataset)
for epoch in range(epochs):
    loss = 0
    rec = np.asarray([])
    for bt in range(int(num_batches_train)):
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = torch.from_numpy(train_dataset[bt*BATCH_SIZE:(bt+1)*BATCH_SIZE]).float()
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = net(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        if epoch == 29:
            rec = np.concatenate((rec,outputs.detach().numpy()))
            
    
    if epoch == 29: 
        plt.plot(rec,'r')
        for i in range(10):
            iri = np.concatenate((iri,np.ones(250)*data['IRI_sequence'][0][i]))
    iri = iri*0.01
    plt.plot(iri,'g')
    plt.legend(['actual','reconstructed','iri'])
    plt.title('Train')
    # compute the epoch training loss
    loss = loss / len(train_dataset)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


plt.figure()
plt.plot(test_dataset)
num_batches_train = len(test_dataset)/BATCH_SIZE
rec = np.asarray([])
for bt in range(int(num_batches_train)):
    # reshape mini-batch data to [N, 784] matrix
    # load it to the active device
    batch_features = torch.from_numpy(test_dataset[bt*BATCH_SIZE:(bt+1)*BATCH_SIZE]).float()
    
    # reset the gradients back to zero
    # PyTorch accumulates gradients on subsequent backward passes
    optimizer.zero_grad()
    
    # compute reconstructions
    outputs = net(batch_features)
    
    # compute training reconstruction loss
    train_loss = criterion(outputs, batch_features)
    
    # compute accumulated gradients
    train_loss.backward()
    
    # perform parameter update based on current gradients
    optimizer.step()
    
    # add the mini-batch training loss to epoch loss
    loss += train_loss.item()
    rec = np.concatenate((rec,outputs.detach().numpy()))

# compute the epoch training loss
loss = loss / len(test_dataset)
plt.plot(rec,'r')
plt.legend(['actual','reconstructed'])
plt.title('Test')

# display the epoch training loss
print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
#%%
def resample(seq, to_length, window_size):
    '''
    Resample a sequence/
    Parameters
    ----------
    seq : np.array
        Sequence to be resampled.
    to_length : int
        Resample to this number of points.
    Returns
    -------
    d_resampled : np.array
        resampled distance (0,10)
    y_resampled : np.array
        resampled input sequence.
    '''
    # Downsample if needed
    seq_len = seq.shape[0]
    if seq_len>to_length:
        seq = choice(seq, to_length)
        seq_len = seq.shape[0] #
    # Current
    d = np.linspace(0, window_size, seq_len)
    f = interpolate.interp1d(d, seq)
    # Generate new points
    d_new = np.random.uniform(low=0, high=d[-1], size=(to_length - seq_len))
    # Append new to the initial
    d_resampled = sorted(np.concatenate((d, d_new)))
    # Estimate y at points
    y_resampled = f(d_resampled)
    return d_resampled, y_resampled
def resample_df(df, feats_to_resample, to_lengths_dict = {}, window_size = None):
    input_feats_resampled = []
    # Filter rows with less than 2 points (can't resample those)
    for feat in feats_to_resample:
        df[feat+'_len'] =  df[feat].apply(lambda seq: 1 if isinstance(seq, float) else seq.shape[0])
        df.mask(df[feat+'_len']<2, inplace = True)
    # Drop nans (rows with NaN/len<2) and reset index
    df.dropna(subset = feats_to_resample, inplace = True)
    df.reset_index(drop = True, inplace = True)
    # Resample to the maximum
    for feat in feats_to_resample:
        print('Resampling feature: ',feat)
        #max_len = max(df[feat].apply(lambda seq: seq.shape[0]))
        to_length = to_lengths_dict[feat]
        new_feats_resampled = ['{0}_d_resampled'.format(feat), '{0}_resampled'.format(feat)]
        df[new_feats_resampled ] = df.apply(lambda seq: resample(seq[feat], to_length = to_length, window_size = window_size),
                                        axis=1, result_type="expand")
        input_feats_resampled.append('{0}_resampled'.format(feat))
    return df,  input_feats_resampled

# Resample length
                if chunk==0:
                    for feat in input_feats:
                        a = df_chunk[feat].apply(lambda seq: seq.shape[0])
                        l = int(a.quantile(0.90))
                        to_lengths_dict[feat] = l
                        print(to_lengths_dict)
                        #to_lengths_dict = {'GM.acc.xyz.z': 369, 'GM.obd.spd_veh.value':309} # this was used for motorway
                # Resample chunk df
                df_chunk, feats_resampled = resample_df(df_chunk, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)