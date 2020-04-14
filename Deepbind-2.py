#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import keras
import tensorflow as tf

def convert_model(file_name, input_len, window_size=0):
    
    data = [x.rstrip() for x in open(file_name).read().split("\n")]
    
    reverse_complement = int(data[1].split(" = ")[1])
    num_detectors = int(data[2].split(" = ")[1])
    detector_len = int(data[3].split(" = ")[1])
    has_avg_pooling = int(data[4].split(" = ")[1])
    num_hidden = int(data[5].split(" = ")[1])
    
    if (window_size < 1):
        window_size = (int)(detector_len*1.5) #copying deepbind code
    if (window_size > input_len):
        window_size = input_len

    detectors = (np.array(
        [float(x) for x in data[6].split(" = ")[1].split(",")])
        .reshape(detector_len, 4, num_detectors))
    biases = np.array([float(x) for x in data[7].split(" = ")[1].split(",")])
    weights1 = np.array([float(x) for x in data[8].split(" = ")[1].split(",")]).reshape(
                        num_detectors*(2 if has_avg_pooling else 1),
                        (1 if num_hidden==0 else num_hidden))
    if (has_avg_pooling > 0):
        #in the orignal deepbind model, these weights are interleaved.
        #what a nightmare.
        weights1 = weights1.reshape((num_detectors,2,-1))
        new_weights1 = np.zeros((2*num_detectors, weights1.shape[-1]))
        new_weights1[:num_detectors, :] = weights1[:,0,:]
        new_weights1[num_detectors:, :] = weights1[:,1,:]
        weights1 = new_weights1
    biases1 = np.array([float(x) for x in data[9].split(" = ")[1].split(",")]).reshape(
                        (1 if num_hidden==0 else num_hidden))
    if (num_hidden > 0):
        #print("Model has a hidden layer")
        weights2 = np.array([float(x) for x in data[10].split(" = ")[1].split(",")]).reshape(
                        num_hidden,1)
        biases2 = np.array([float(x) for x in data[11].split(" = ")[1].split(",")]).reshape(
                        1)
    
    
    def seq_padding(x):
        return tf.pad(x,
                [[0, 0],
                 [detector_len-1, detector_len-1],
                 [0, 0]],
                mode='CONSTANT',
                name=None,
                constant_values=0.25)

    
    input_tensor = keras.layers.Input(shape=(None,4))
    padding_out_fwd = keras.layers.Lambda(seq_padding)(input_tensor)
    conv_layer = keras.layers.Conv1D(filters=num_detectors,
                                  kernel_size=detector_len,
                                  activation="relu")
    conv_out_fwd = conv_layer(padding_out_fwd)
    pool_out_fwd = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_fwd)
    if (has_avg_pooling > 0):
        #print("Model has average pooling")
        gap_out_fwd = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_fwd)
        pool_out_fwd = keras.layers.Concatenate(axis=-1)([pool_out_fwd, gap_out_fwd])        
    dense1_layer = keras.layers.Dense((1 if num_hidden==0 else num_hidden))
    dense1_out_fwd = keras.layers.TimeDistributed(dense1_layer)(pool_out_fwd)
    if (num_hidden > 0):
        dense1_out_fwd = keras.layers.Activation("relu")(dense1_out_fwd)
        dense2_layer = keras.layers.Dense(1)
        dense2_out_fwd = keras.layers.TimeDistributed(dense2_layer)(dense1_out_fwd)
    
    if (reverse_complement > 0):
        #print("Model has reverse complementation")
        padding_out_rev = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(padding_out_fwd)
        conv_out_rev = conv_layer(padding_out_rev)
        pool_out_rev = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_rev)
        if (has_avg_pooling > 0):
            #print("Model has average pooling")
            gap_out_rev = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_rev)
            pool_out_rev = keras.layers.Concatenate(axis=-1)([pool_out_rev, gap_out_rev])
        dense1_out_rev = keras.layers.TimeDistributed(dense1_layer)(pool_out_rev)
        if (num_hidden > 0):
            dense1_out_rev = keras.layers.Activation("relu")(dense1_out_rev)
            dense2_out_rev = keras.layers.TimeDistributed(dense2_layer)(dense1_out_rev)
    
    cross_seq_max = keras.layers.Lambda(lambda x: tf.reduce_max(x,axis=1)[:,0],
                                        output_shape=lambda x: (None,1))
    
    if (reverse_complement > 0):
        if (num_hidden > 0):
            max_fwd = cross_seq_max(dense2_out_fwd)
            max_rev = cross_seq_max(dense2_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
        else:
            max_fwd = cross_seq_max(dense1_out_fwd)
            max_rev = cross_seq_max(dense1_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
    else:
        if (num_hidden > 0):
            output = cross_seq_max(dense2_out_fwd)
        else:
            output = cross_seq_max(dense1_out_fwd)
        
    #print(input_tensor)   
    model = keras.models.Model(inputs = [input_tensor],
                               outputs = [output])
    model.compile(loss="mse", optimizer="adam")
    conv_layer.set_weights([detectors, biases])
    dense1_layer.set_weights([weights1, biases1])
    if (num_hidden > 0):
        dense2_layer.set_weights([weights2, biases2])
        
    return model

def onehot_encode_sequences(sequences):
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            arr[i, mapping[letter]] = 1.0
        onehot.append(arr)
    return onehot


# In[2]:


import glob
import os
import re 

####Reading sequences from FASTA txt files and converting them into a list 

file_list = glob.glob(os.path.join(os.getcwd(), "./la", "*.txt"))

new = []
dna = []
sequences = []

for file_path in file_list:
    with open(file_path) as f_input:
        seq = ''        
        #forloop through the lines
        for line in f_input: 
            header = re.search(r'^>\w+', line)
            #if line contains the header '>' then append it to the dna list 
            if header:
                line = line.rstrip("\n")
                dna.append(line)            
            # in the else statement is where I have problems, what I would like is
            #else: 
                #the proceeding lines before the next '>' is the sequence for each header,
                #concatenate these lines into one string and append to the sequences list 
            else:               
                seq = line.replace('\n', '')  
                
                sequences.append(seq)
                
#print(sequences)

###Removing empty strings from list of sequences
new_sequences = []
for string in sequences:
    if (string != ""):
        new_sequences.append(string)


# In[3]:


###Restricting the sequence to only 20 characters 

n = 20
final_seq = []    
for i in new_sequences:    
    final_seq.append([i[j:j+n] for j in range(0, len(i), n)])
print (final_seq)

#The output is a list of list


# In[4]:


#converting list of list to list
from itertools import chain
final_seq = list(chain.from_iterable(final_seq))
final_seq


# In[5]:


count = 0 

#### Get the ID files from local directory - change path here 
### Only 20 ID files are taken for testing sake. This part takes a little longer to run when more ID files are inputed. 
### This code works for all sequences length but its best if only 5 sequences are run at a time


path = "./minidb"   
for file_name in glob.glob(os.path.join(path, '*.txt')):
    print("On model", file_name)
    print(count)
    
    
    onehot_sequences = onehot_encode_sequences(final_seq)    
    for i in final_seq:
        model = convert_model(file_name = file_name, input_len= len(i))
    j = 0
    for j in range(len(final_seq)):
        print("\n".join(str(x) for x
                        in model.predict(np.array(onehot_sequences[j:j+1])[:,:,:])))


# In[ ]:




